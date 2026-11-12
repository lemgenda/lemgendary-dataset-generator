"""R4-B7: ``sync pull`` without a whole-dataset archive, selected members, missing root."""

from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import patch

from core.kaggle_file_download import (
    FileDownloadClients,
    download_dataset_files,
    list_dataset_files,
)
from core.kaggle_http import HttpRangeReader, KaggleHttpError, SignedUrl, signed_url
from core.sync_download import DownloadRequest, perform_dataset_download
from core.sync_partial_download import is_missing_archive_error
from tools import manifold_sync
from utils.member_paths import (
    UnsafeMemberPath,
    common_wrapper,
    local_path,
    matches_prefixes,
)
from utils.zip_ranges import extract_members, read_central_directory, select_members

_WRAP = "LemGendizedDemo"


def _fake(**attrs: Any) -> Any:
    """A duck-typed stand-in for a client object."""
    return SimpleNamespace(**attrs)


class _Response:
    """A minimal ``requests.Response`` stand-in."""

    def __init__(self, status: int, content: bytes = b"",
                 headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self.content = content
        self.headers = headers or {}

    def iter_content(self, size: int) -> Any:
        """The body in ``size`` chunks."""
        for at in range(0, len(self.content), size):
            yield self.content[at:at + size]

    def raise_for_status(self) -> None:
        """Raise for an error status, like requests."""
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


class _BytesReader:
    """A RangeReader over bytes that records how much was read."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.read_bytes = 0

    def size(self) -> int:
        """Length of the data."""
        return len(self.data)

    def read(self, start: int, end: int) -> bytes:
        """Bytes ``start`` to ``end`` inclusive."""
        self.read_bytes += end - start + 1
        return self.data[start:end + 1]


def _demo_zip(big: int = 2_000_000) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{_WRAP}/images/train/a.jpg", b"\x01" * big, zipfile.ZIP_STORED)
        zf.writestr(f"{_WRAP}/labels/train/a.txt", "0 0.5 0.5 1 1\n", zipfile.ZIP_DEFLATED)
        zf.writestr(f"{_WRAP}/labels/val/b.txt", "", zipfile.ZIP_DEFLATED)
        zf.writestr(f"{_WRAP}/images/val/b.jpg", b"\x02" * big, zipfile.ZIP_STORED)
        zf.writestr(f"{_WRAP}/classes.txt", "face\n", zipfile.ZIP_DEFLATED)
    return buf.getvalue()


class MemberPathTests(unittest.TestCase):
    """Wrapper folder, --only selection and path safety."""

    def test_wrapper_and_prefixes(self) -> None:
        """The shared top folder is stripped; ``labels/`` selects the folder only."""
        names = [f"{_WRAP}/labels/a.txt", f"{_WRAP}/classes.txt"]
        self.assertEqual(common_wrapper(names), _WRAP)
        self.assertEqual(common_wrapper(["labels/a.txt", "labels/b.txt"]), "")
        self.assertTrue(matches_prefixes("labels/train/a.txt", ("labels/",)))
        self.assertTrue(matches_prefixes("labels/train/a.txt", ("labels",)))
        self.assertTrue(matches_prefixes("classes.txt", ("classes.txt",)))
        self.assertFalse(matches_prefixes("labelsx/a.txt", ("labels",)))

    def test_unsafe_paths_are_refused(self) -> None:
        """A ``..`` or drive part never becomes a local path."""
        for rel in ("../x", "a/../../x", "C:/x", "a//b"):
            with self.assertRaises(UnsafeMemberPath):
                local_path(Path("dest"), rel)


class ZipRangeTests(unittest.TestCase):
    """Selected members come out of a remote zip through byte ranges."""

    def test_only_selected_members_are_fetched_and_verified(self) -> None:
        """labels/ and classes.txt are written; the large images are never read."""
        data = _demo_zip()
        reader = _BytesReader(data)
        members = select_members(read_central_directory(reader), ("labels/", "classes.txt"))
        self.assertEqual(sorted(m.rel for m in members),
                         ["classes.txt", "labels/train/a.txt", "labels/val/b.txt"])
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            stats = extract_members(reader, members, dest)
            self.assertEqual((stats.written, stats.failed), (3, []))
            self.assertEqual((dest / "labels/train/a.txt").read_text(), "0 0.5 0.5 1 1\n")
            self.assertEqual((dest / "classes.txt").read_text(), "face\n")
            self.assertLess(reader.read_bytes, len(data) // 4)
            again = extract_members(reader, members, dest)
            self.assertEqual((again.written, again.skipped), (0, 3))

    def test_corrupt_member_is_not_written(self) -> None:
        """A member whose CRC does not match is reported and left out."""
        data = bytearray(_demo_zip(big=1000))
        at = data.find(b"0 0.5")
        if at < 0:
            entries = read_central_directory(_BytesReader(bytes(data)))
            label = next(e for e in entries if e.name.endswith("a.txt"))
            at = label.offset + 30 + len(label.name) + 2
        data[at] ^= 0xFF
        reader = _BytesReader(bytes(data))
        members = select_members(read_central_directory(reader), ("labels/train/",))
        with tempfile.TemporaryDirectory() as tmp:
            stats = extract_members(reader, members, Path(tmp))
            self.assertEqual(stats.failed, ["labels/train/a.txt"])
            self.assertFalse((Path(tmp) / "labels/train/a.txt").exists())


def _range_answer(data: bytes, rng: str) -> _Response:
    start_s, _, end_s = rng.removeprefix("bytes=").partition("-")
    start = int(start_s)
    end = int(end_s) if end_s else len(data) - 1
    return _Response(206, data[start:end + 1],
                     {"Content-Range": f"bytes {start}-{end}/{len(data)}"})


def _http(data: bytes, fail: int = 0) -> Any:
    """Fake storage client: answers range requests from ``data`` after ``fail`` rejections."""
    state = SimpleNamespace(fail=fail, urls=[])

    def get(url: str, headers: dict[str, str] | None = None, **_kw: Any) -> _Response:
        state.urls.append(url)
        if state.fail:
            state.fail -= 1
            return _Response(403)
        rng = (headers or {}).get("Range", "")
        return _range_answer(data, rng) if rng else _Response(200, data)

    state.get = get
    return state


class HttpRangeReaderTests(unittest.TestCase):
    """Signed URLs are refreshed when storage rejects them."""

    def test_rejected_url_is_refreshed(self) -> None:
        """A 403 resolves a new URL and the range is fetched again."""
        urls = iter(["u1", "u2"])
        signed = SignedUrl(lambda: next(urls))
        http = _http(b"0123456789", fail=1)
        reader = HttpRangeReader(signed, http, sleep=lambda _s: None)
        self.assertEqual(reader.read(2, 4), b"234")
        self.assertEqual(http.urls, ["u1", "u2"])
        self.assertEqual(reader.size(), 10)

    def test_signed_url_404_means_no_archive(self) -> None:
        """Kaggle's 404 for the whole-dataset download is None, a redirect is the URL."""
        session = _fake(get=lambda *_a, **_k: _Response(404))
        self.assertIsNone(signed_url(session, "o/s"))
        session = _fake(get=lambda *_a, **_k: _Response(302, headers={"Location": "L"}))
        self.assertEqual(signed_url(session, "o/s"), "L")
        session = _fake(get=lambda *_a, **_k: _Response(500))
        with self.assertRaises(KaggleHttpError):
            signed_url(session, "o/s")


def _listing(pages: list[list[tuple[str, int]]]) -> Any:
    def dataset_list_files(_handle: str, page_token: str | None = None,
                           page_size: int = 20) -> Any:
        del page_size
        index = int(page_token or 0)
        files = [SimpleNamespace(name=n, total_bytes=s) for n, s in pages[index]]
        token = str(index + 1) if index + 1 < len(pages) else ""
        return SimpleNamespace(files=files, next_page_token=token)

    return SimpleNamespace(dataset_list_files=dataset_list_files)


def _storage(payloads: dict[str, bytes]) -> Any:
    """Fake signed-URL storage: file name -> payload, recording (url, range) requests."""
    state = SimpleNamespace(payloads=payloads, requests=[])

    def get(url: str, headers: dict[str, str] | None = None, **_kw: Any) -> _Response:
        rng = (headers or {}).get("Range", "")
        state.requests.append((url, rng))
        body = state.payloads[url]
        if rng:
            return _Response(206, body[int(rng.removeprefix("bytes=").rstrip("-")):])
        return _Response(200, body)

    state.get = get
    return state


def _kaggle_session() -> Any:
    def get(url: str, **_kw: Any) -> _Response:
        name = url.rsplit("/", 1)[1].replace("%2F", "/")
        return _Response(302, headers={"Location": name})

    return SimpleNamespace(get=get)


def _zipped(name: str, body: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, body)
    return buf.getvalue()


class FileByFileTests(unittest.TestCase):
    """Datasets without a whole-dataset archive are downloaded file by file."""

    def _clients(self, api: Any, storage: Any) -> FileDownloadClients:
        return FileDownloadClients(api=api, session=_kaggle_session(), http=storage,
                                   sleep=lambda _s: None, workers=2)

    def test_all_pages_listed(self) -> None:
        """Every page of the listing is read."""
        api = _listing([[("a", 1), ("b", 2)], [("c", 3)]])
        self.assertEqual([f.name for f in list_dataset_files(api, "o/s")], ["a", "b", "c"])

    def test_download_verify_skip_resume_and_unzip(self) -> None:
        """Files land without the wrapper, complete ones are skipped, .part resumes."""
        names = {f"{_WRAP}/shards/train/shard-00000.data": b"S" * 50,
                 f"{_WRAP}/labels/a.txt": b"0 1 1 1 1",
                 f"{_WRAP}/classes.txt": b"face\n"}
        storage = _storage(dict(names))
        storage.payloads[f"{_WRAP}/classes.txt"] = _zipped("classes.txt", b"face\n")
        api = _listing([[(n, len(b)) for n, b in names.items()]])
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            (dest / "labels").mkdir()
            (dest / "labels/a.txt").write_bytes(b"0 1 1 1 1")
            shard = dest / "shards/train/shard-00000.data"
            shard.parent.mkdir(parents=True)
            shard.with_name(shard.name + ".part").write_bytes(b"S" * 20)
            result = download_dataset_files("o/s", dest, self._clients(api, storage))
            self.assertTrue(result.ok, result.failed)
            self.assertEqual((result.downloaded, result.skipped), (2, 1))
            self.assertEqual(shard.read_bytes(), b"S" * 50)
            self.assertEqual((dest / "classes.txt").read_bytes(), b"face\n")
            self.assertIn((f"{_WRAP}/shards/train/shard-00000.data", "bytes=20-"),
                          storage.requests)

    def test_size_mismatch_fails_and_only_filters(self) -> None:
        """A file shorter than listed fails; --only limits the listing."""
        storage = _storage({f"{_WRAP}/labels/a.txt": b"short", f"{_WRAP}/x.bin": b"x"})
        api = _listing([[(f"{_WRAP}/labels/a.txt", 99), (f"{_WRAP}/x.bin", 1)]])
        with tempfile.TemporaryDirectory() as tmp:
            result = download_dataset_files("o/s", Path(tmp), self._clients(api, storage),
                                            ("labels/",))
            self.assertEqual(list(result.failed), [f"{_WRAP}/labels/a.txt"])
            self.assertFalse((Path(tmp) / "x.bin").exists())


class FallbackTests(unittest.TestCase):
    """The whole-dataset 404 switches to the file-by-file download."""

    def test_missing_archive_error_detection(self) -> None:
        """A 404 response or Kaggle's "No gcs url found" text is a missing archive."""
        self.assertTrue(is_missing_archive_error(OSError("No gcs url found")))
        err: Any = OSError("x")
        err.response = SimpleNamespace(status_code=404)
        self.assertTrue(is_missing_archive_error(err))
        self.assertFalse(is_missing_archive_error(OSError("timeout")))

    def test_api_404_downloads_file_by_file(self) -> None:
        """perform_dataset_download falls back instead of trying the CLI and kagglehub."""
        api = SimpleNamespace(authenticate=lambda: None, dataset_download_files=None)

        def _no_zip(*_a: Any, **_k: Any) -> None:
            raise OSError("404 Client Error: No gcs url found")

        api.dataset_download_files = _no_zip
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "LemGendizedDemo"
            request = DownloadRequest(clean_repo_id="o/demo", target_dir=target,
                                      root_datasets_dir=root, manifold_name="LemGendizedDemo",
                                      slug="demo")

            def _files(_handle: str, dest: Path, _prefixes: tuple[str, ...] = ()) -> bool:
                (dest / "index.json").write_text("[]", encoding="utf-8")
                return True

            with patch("core.sync_download.load_kaggle_api", return_value=lambda: api), \
                    patch("core.sync_download.download_file_by_file", side_effect=_files) as f, \
                    patch("core.sync_download.subprocess.run") as cli:
                self.assertTrue(perform_dataset_download(request))
            f.assert_called_once()
            cli.assert_not_called()

    def test_only_goes_to_selected_download(self) -> None:
        """A request with ``only`` never downloads the whole archive."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = DownloadRequest(clean_repo_id="o/demo", target_dir=root / "D",
                                      root_datasets_dir=root, manifold_name="D", slug="demo")
            with patch("core.sync_download.download_selected", return_value=True) as sel, \
                    patch("core.sync_download._fetch_remote_archive") as whole:
                self.assertTrue(perform_dataset_download(request, only=("labels/",)))
            sel.assert_called_once_with("o/demo", root / "D", ("labels/",))
            whole.assert_not_called()


class MissingRootTests(unittest.TestCase):
    """``sync pull`` creates a missing datasets root and measures its nearest parent."""

    def test_missing_root_is_created(self) -> None:
        """disk_usage is asked about an existing folder and the root exists afterwards."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "missing" / "LemGendaryDatasets"
            probed: list[Path] = []
            real = manifold_sync.shutil.disk_usage

            def _usage(path: Path) -> Any:
                probed.append(Path(path))
                return real(path)

            with patch.object(manifold_sync, "OUT_PARENT", root), \
                    patch.object(manifold_sync.shutil, "disk_usage", side_effect=_usage), \
                    patch.object(manifold_sync, "setup_auth"), \
                    patch.object(manifold_sync, "perform_dataset_download",
                                 return_value=True) as download:
                manifold_sync.action_get("o/demo", output_name="LemGendizedDemo",
                                         only=("labels/",))
            self.assertEqual(probed, [Path(tmp)])
            self.assertTrue(root.is_dir())
            self.assertEqual(download.call_args.kwargs["only"], ("labels/",))


if __name__ == "__main__":
    unittest.main()
