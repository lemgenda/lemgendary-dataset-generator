"""Kaggle dataset pull/push fixes from the Kaggle audit (K-DS-1 to K-DS-8).

Every Kaggle call is replaced by a fake; nothing reaches the network.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from api.routes.kaggle import get_kaggle_status
from cli import app as cli_app
from core import kaggle_credentials as creds
from core.cli_args import __version__
from core.docs.context import _resolve_citations
from core.docs.context_types import SpecInputs
from core.docs.generators.dataset_info_yaml import generate_dataset_info_yaml
from core.docs.generators.kaggle_metadata import _describe_directory
from core.docs.generators.readme_lineage import ref_url
from core.docs.generators.readme_md import generate_readme_md
from core.docs.lineage_tally import tally_sources
from core.docs.metadata import UNIFIED_DATA
from core.paths import UNIFIED_DATA as UNIFIED_DATA_FILE
from core.sync_auth import setup_kaggle_auth
from core.sync_download import DownloadRequest, perform_dataset_download
from core.sync_metadata import _list_remote_files
from core.sync_metadata_settings import build_dataset_settings
from core.sync_remote import ensure_kaggle_dataset_exists
from core.sync_upload import perform_dataset_upload
from core.sync_version_notes import build_version_notes
from sources.kaggle import KaggleRequest, run_kaggle_action

_KAGGLE_ENV = ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME", "KAGGLE_CONFIG_DIR")


def _clean_env() -> dict[str, str]:
    """The current environment without any Kaggle credential variable."""
    return {k: v for k, v in os.environ.items() if k not in _KAGGLE_ENV}


class _TempCase(unittest.TestCase):
    """A fresh temp folder per test, used as the home folder for Kaggle files."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="kaggle_sync_audit_"))
        self.addCleanup(shutil.rmtree, self.root)
        self.home = self.root / "home"
        self.home.mkdir()
        self.enterContext(patch.object(Path, "home", return_value=self.home))
        self.enterContext(patch.dict(os.environ, _clean_env(), clear=True))


class KaggleCredentialTests(_TempCase):
    """KGAT_ tokens stay access tokens; legacy keys stay keys (K-AUTH, datasets side)."""

    def test_access_token_goes_to_api_token_unchanged(self) -> None:
        """A KGAT_ token is exported as KAGGLE_API_TOKEN with its prefix, never as a key."""
        creds.export_credential(creds.KaggleCredential("KGAT_abc", None, "test"), "owner")
        self.assertEqual(os.environ["KAGGLE_API_TOKEN"], "KGAT_abc")
        self.assertNotIn("KAGGLE_KEY", os.environ)
        self.assertEqual(os.environ["KAGGLE_USERNAME"], "owner")

    def test_legacy_key_goes_to_key_with_its_account(self) -> None:
        """A legacy key is exported as KAGGLE_KEY with the vault account, not the owner."""
        creds.export_credential(creds.KaggleCredential("abc123", "keyowner", "test"), "owner")
        self.assertEqual(os.environ["KAGGLE_KEY"], "abc123")
        self.assertNotIn("KAGGLE_API_TOKEN", os.environ)
        self.assertEqual(os.environ["KAGGLE_USERNAME"], "keyowner")

    def test_setup_exports_the_stored_token(self) -> None:
        """setup_kaggle_auth exports the vault token by kind and reports success."""
        stored = creds.KaggleCredential("KGAT_vault", "lem", creds.SOURCE_VAULT)
        with patch("core.sync_auth.stored_credential", return_value=stored):
            self.assertTrue(setup_kaggle_auth(default_user="owner"))
        self.assertEqual(os.environ["KAGGLE_API_TOKEN"], "KGAT_vault")
        self.assertNotIn("KAGGLE_KEY", os.environ)

    def test_setup_without_any_secret_is_false(self) -> None:
        """No stored secret, no environment and no ~/.kaggle file: no credentials."""
        with patch("core.sync_auth.stored_credential", return_value=None):
            self.assertFalse(setup_kaggle_auth(default_user="owner"))

    def test_access_token_file_counts(self) -> None:
        """~/.kaggle/access_token (the kagglesdk file) is a credential source."""
        (self.home / ".kaggle").mkdir()
        (self.home / ".kaggle" / "access_token").write_text("KGAT_file\n", encoding="utf-8")
        with patch("core.sync_auth.stored_credential", return_value=None):
            self.assertTrue(setup_kaggle_auth(default_user="owner"))
        with patch("core.kaggle_credentials.get_kaggle_vault_credentials",
                   return_value=(None, None)):
            status = asyncio.run(get_kaggle_status())
        self.assertTrue(status["authenticated"])
        self.assertTrue(status["auth_methods"]["user_access_token"])
        self.assertFalse(status["auth_methods"]["user_kaggle_json"])

    def test_action_without_credentials_fails_cleanly(self) -> None:
        """The CLI action returns 1 instead of letting the Kaggle client exit the process."""
        with patch("sources.kaggle.setup_kaggle_auth", return_value=False), \
                patch("sources.kaggle.track_kaggle_dataset_status") as track:
            code = run_kaggle_action(KaggleRequest(action="status", repo_id="owner/slug"))
        self.assertEqual(code, 1)
        track.assert_not_called()


def _paged_listing(pages: list[Any]) -> SimpleNamespace:
    """Fake KaggleApi with ``dataset_list_files`` serving ``pages`` in order.

    ``tokens`` records the page token of every request; an exception in
    ``pages`` is raised instead of returned.
    """
    tokens: list[str | None] = []

    def dataset_list_files(_handle: str, page_token: str | None = None,
                           page_size: int = 20) -> Any:
        del page_size
        tokens.append(page_token)
        page = pages[len(tokens) - 1]
        if isinstance(page, Exception):
            raise page
        return page

    return SimpleNamespace(dataset_list_files=dataset_list_files, tokens=tokens)


def _page(names: list[str], token: str = "") -> SimpleNamespace:
    return SimpleNamespace(files=[SimpleNamespace(name=n) for n in names], next_page_token=token)


class RemoteFileListingTests(unittest.TestCase):
    """K-DS-1: column descriptors see every remote file, not only page one."""

    def test_reads_every_page(self) -> None:
        """Pages are requested with the previous next_page_token until it is empty."""
        api = _paged_listing([_page(["a", "b"], "t1"), _page(["c"], "t2"), _page(["d"])])
        self.assertEqual(_list_remote_files(api, "o/s"), {"a", "b", "c", "d"})
        self.assertEqual(api.tokens, [None, "t1", "t2"])

    def test_repeated_token_stops(self) -> None:
        """A token Kaggle sends twice ends the listing instead of looping."""
        api = _paged_listing([_page(["a"], "t1"), _page(["b"], "t1")])
        with self.assertLogs("core.sync_metadata", logging.WARNING):
            self.assertEqual(_list_remote_files(api, "o/s"), {"a", "b"})

    def test_error_keeps_names_read(self) -> None:
        """A failed page is logged and the names read so far are kept."""
        api = _paged_listing([_page(["a"], "t1"), OSError("boom")])
        with self.assertLogs("core.sync_metadata", logging.WARNING):
            self.assertEqual(_list_remote_files(api, "o/s"), {"a"})


class ExactArchiveTests(unittest.TestCase):
    """K-DS-2: a download only ever unpacks its own archive."""

    def test_foreign_zip_is_never_used(self) -> None:
        """Another dataset's zip in the root is left alone and the download fails."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "LemGendizedDemo"
            target.mkdir()
            (target / "old.txt").write_text("stale", encoding="utf-8")
            foreign = root / "otherdataset.zip"
            foreign.write_bytes(b"PK")
            request = DownloadRequest(
                clean_repo_id="owner/demo", target_dir=target, root_datasets_dir=root,
                manifold_name="LemGendizedDemo", slug="demo",
            )
            with patch("core.sync_download._download_via_api_or_cli", return_value=True), \
                    patch("core.sync_download._unpack_archive") as unpack:
                self.assertFalse(perform_dataset_download(request))
            unpack.assert_not_called()
            self.assertTrue(foreign.exists())

    def test_own_zip_is_unpacked(self) -> None:
        """The archive Kaggle names <slug>.zip is the one unpacked."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "LemGendizedDemo"
            own = root / "demo.zip"
            request = DownloadRequest(
                clean_repo_id="owner/demo", target_dir=target, root_datasets_dir=root,
                manifold_name="LemGendizedDemo", slug="demo",
            )

            def _download(_req: DownloadRequest) -> bool:
                own.write_bytes(b"PK")
                return True

            with patch("core.sync_download._download_via_api_or_cli", side_effect=_download), \
                    patch("core.sync_download._unpack_archive", return_value=True) as unpack:
                self.assertTrue(perform_dataset_download(request))
            self.assertEqual(unpack.call_args.args[0], own)


def _write_index(manifold: Path, splits: dict[str, int]) -> None:
    entries = [{"name": f"{split}{i}", "split": split}
               for split, n in splits.items() for i in range(n)]
    (manifold / "index.json").write_text(json.dumps(entries), encoding="utf-8")


class VersionNotesTests(unittest.TestCase):
    """K-DS-3: every pushed version carries notes (compiler release, real counts)."""

    def test_notes_from_index(self) -> None:
        """Counts come from index.json per split."""
        with tempfile.TemporaryDirectory() as tmp:
            manifold = Path(tmp)
            _write_index(manifold, {"train": 3, "val": 1})
            notes = build_version_notes(manifold)
        self.assertEqual(
            notes, f"LemGendary Dataset Compiler v{__version__}; 4 samples (train 3, val 1)"
        )

    def test_fake_push_sends_version_notes(self) -> None:
        """A full upload against a fake Kaggle passes the notes to kagglehub."""
        with tempfile.TemporaryDirectory() as tmp:
            manifold = Path(tmp) / "LemGendizedDemo"
            manifold.mkdir()
            _write_index(manifold, {"train": 2, "val": 1})
            with patch("core.sync_upload.ensure_kaggle_dataset_exists", return_value=True), \
                    patch("core.sync_upload.get_dataset_version_info",
                          return_value={"latest_version": 4}), \
                    patch("core.sync_upload.kagglehub.dataset_upload") as upload, \
                    patch("core.sync_upload.push_kaggle_dataset_metadata") as meta:
                self.assertTrue(perform_dataset_upload(manifold, "owner/demo", no_wait=True))
        upload.assert_called_once()
        self.assertEqual(upload.call_args.args[0], "owner/demo")
        self.assertEqual(
            upload.call_args.kwargs["version_notes"],
            f"LemGendary Dataset Compiler v{__version__}; 3 samples (train 2, val 1)",
        )
        meta.assert_not_called()

    def test_extra_notes_follow_the_generated_part(self) -> None:
        """Text given with --notes follows the release and counts; blank text adds nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            manifold = Path(tmp)
            _write_index(manifold, {"train": 1})
            notes = build_version_notes(manifold, "  Masks rebuilt;\n split by file id. ")
            blank = build_version_notes(manifold, "   ")
        base = f"LemGendary Dataset Compiler v{__version__}; 1 samples (train 1)"
        self.assertEqual(notes, f"{base}. Masks rebuilt; split by file id.")
        self.assertEqual(blank, base)

    def test_cli_push_passes_notes_to_kagglehub(self) -> None:
        """`sync push --notes` reaches kagglehub's version_notes through the push path."""
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            manifold = parent / "LemGendizedDemo"
            manifold.mkdir()
            _write_index(manifold, {"train": 2})
            with patch("tools.manifold_sync.OUT_PARENT", parent), \
                    patch("tools.manifold_sync.setup_auth", return_value=True), \
                    patch("tools.manifold_sync.cleanup_temp_archives"), \
                    patch("core.sync_upload.ensure_kaggle_dataset_exists", return_value=True), \
                    patch("core.sync_upload.get_dataset_version_info",
                          return_value={"latest_version": 1}), \
                    patch("core.sync_upload.kagglehub.dataset_upload") as upload:
                result = CliRunner().invoke(cli_app, [
                    "sync", "push", "--model", "LemGendizedDemo", "--url", "owner/demo",
                    "--no-wait", "--notes", "Labels rebuilt.",
                ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            upload.call_args.kwargs["version_notes"],
            f"LemGendary Dataset Compiler v{__version__}; 2 samples (train 2). Labels rebuilt.",
        )


class _FakeKaggleApi:
    """Records the dataset_create_new call and the placeholder metadata it was given."""

    calls: ClassVar[list[dict[str, Any]]] = []

    def authenticate(self) -> None:
        """No credentials are needed by the fake."""

    def dataset_create_new(self, **kwargs: Any) -> SimpleNamespace:
        """Capture the arguments and the folder's dataset-metadata.json."""
        folder = Path(kwargs["folder"])
        meta = json.loads((folder / "dataset-metadata.json").read_text(encoding="utf-8"))
        _FakeKaggleApi.calls.append({**kwargs, "meta": meta})
        return SimpleNamespace(status="ok")


class PrivatePlaceholderTests(unittest.TestCase):
    """R4-B5: a new dataset starts private with the manifold license."""

    def setUp(self) -> None:
        _FakeKaggleApi.calls = []
        self.enterContext(
            patch("core.sync_remote.load_kaggle_api", return_value=_FakeKaggleApi)
        )
        self.enterContext(
            patch("core.sync_remote.get_dataset_version_info",
                  return_value={"latest_version": 0})
        )

    def test_placeholder_is_private_with_given_license(self) -> None:
        """The create call is never public and carries the registry license."""
        self.assertTrue(ensure_kaggle_dataset_exists(
            "owner/demo", title="Demo Title", license_name="CC-BY-NC-4.0"
        ))
        self.assertEqual(len(_FakeKaggleApi.calls), 1)
        call = _FakeKaggleApi.calls[0]
        self.assertFalse(call["public"])
        self.assertTrue(call["meta"]["isPrivate"])
        self.assertEqual(call["meta"]["licenses"], [{"name": "CC-BY-NC-4.0"}])

    def test_existing_dataset_is_not_recreated(self) -> None:
        """A dataset that already has a version is left alone."""
        with patch("core.sync_remote.get_dataset_version_info",
                   return_value={"latest_version": 3}):
            self.assertTrue(ensure_kaggle_dataset_exists("owner/demo", title="Demo Title"))
        self.assertEqual(_FakeKaggleApi.calls, [])

    def test_metadata_push_applies_registry_visibility(self) -> None:
        """The metadata push sets the stated visibility; unstated stays private."""
        self.assertFalse(build_dataset_settings({"isPrivate": False}, "demo").is_private)
        self.assertTrue(build_dataset_settings({"isPrivate": True}, "demo").is_private)
        self.assertTrue(build_dataset_settings({}, "demo").is_private)

    def test_metadata_push_applies_registry_license(self) -> None:
        """The metadata push replaces the placeholder license with the declared one."""
        settings = build_dataset_settings({"licenses": [{"name": "CC-BY-NC-4.0"}]}, "demo")
        self.assertEqual(
            [lic.name for lic in settings.licenses or [] if lic is not None],
            ["Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"],
        )


class CitationTests(unittest.TestCase):
    """K-DS-5: citations name the training suite repository, not a guessed URL."""

    def test_registry_url_first_then_entry_references(self) -> None:
        """The registry citation_url leads, followed by the entry's papers."""
        spec = SpecInputs(unified_entry={"citations": ["https://arxiv.org/abs/1"]})
        citations = _resolve_citations(spec)
        self.assertEqual(citations[0], UNIFIED_DATA["_registry_metadata"]["citation_url"])
        self.assertEqual(citations[1:], ["https://arxiv.org/abs/1"])

    def test_registry_has_no_model_training_url(self) -> None:
        """The wrong github.com/<kaggle owner>/model-training URL is gone."""
        self.assertNotIn("/model-training", UNIFIED_DATA_FILE.read_text(encoding="utf-8"))


def _write_shard(path: Path, members: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        for name in members:
            info = tarfile.TarInfo(name)
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))


class PublishedMetadataTests(unittest.TestCase):
    """K-DS-6: published metadata follows the data on disk."""

    def test_dataset_info_has_no_path(self) -> None:
        """dataset_info.yaml no longer records the build machine's folder."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "LemGendizedCodeFormer"
            root.mkdir()
            info = yaml.safe_load(generate_dataset_info_yaml(root).read_text(encoding="utf-8"))
        self.assertNotIn("path", info)

    def test_self_tagged_index_lists_sources_without_counts(self) -> None:
        """Real split totals once; provenance sources carry no invented counts."""
        items = [
            {"name": f"s{i}", "source": "LemGendizedCodeFormer-manifold",
             "split": "val" if i < 3 else "train"}
            for i in range(10)
        ]
        tally = tally_sources(
            Path("."), "restoration", "LemGendizedCodeFormer",
            {"provenance_sources": ["A", "B"]}, items,
        )
        self.assertEqual((tally.splits.train, tally.splits.val), (7, 3))
        self.assertEqual(tally.total_samples, 10)
        for counts in tally.sources.values():
            self.assertEqual(set(counts.values()), {None})

    def test_sample_keys_give_exact_source_counts(self) -> None:
        """``<Prefix>_<source>_<idx>`` keys yield exact per-source counts."""
        items = [
            {"name": "CodeFormer_ffhq_000000001", "source": "LemGendizedCodeFormer-manifold",
             "split": "train"},
            {"name": "CodeFormer_ffhq_000000002", "source": "LemGendizedCodeFormer-manifold",
             "split": "val"},
            {"name": "CodeFormer_celeba_000000003",
             "source": "LemGendizedCodeFormer-manifold", "split": "train"},
        ]
        tally = tally_sources(
            Path("."), "restoration", "LemGendizedCodeFormer",
            {"provenance_sources": ["A", "B"]}, items,
        )
        rows = sorted(tally.sources.values(), key=lambda c: -(c["total"] or 0))
        self.assertEqual(rows, [
            {"train": 1, "val": 1, "total": 2},
            {"train": 1, "val": 0, "total": 1},
        ])

    def test_split_named_keys_are_not_sources(self) -> None:
        """A ``<Prefix>_train_<idx>`` key names a split, so no counts are derived."""
        items = [{"name": "X_train_000001", "source": "LemGendizedX-manifold"}]
        tally = tally_sources(Path("."), "quality", "LemGendizedX", {"provenance_sources": ["A"]},
                              items)
        self.assertEqual(tally.sources, {"A": {"train": None, "val": None, "total": None}})

    def test_unrecorded_splits_render_as_not_recorded(self) -> None:
        """Without an index the README states the splits are not recorded, never a guess."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "LemGendizedCodeFormer"
            root.mkdir()
            (root / "dataset_info.yaml").write_text("count: 22000\n", encoding="utf-8")
            _, text = generate_readme_md(root)
        self.assertIn("| Train | not recorded |", text)
        self.assertIn("| Total | 22,000 |", text)
        self.assertNotIn("19,800", text)
        self.assertNotIn("9,900", text)

    def test_readme_reports_real_splits_once_and_lists_sources(self) -> None:
        """Split counts appear once; sources are listed with links and the license note."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "LemGendizedCodeFormer"
            root.mkdir()
            index = [
                {"name": f"s{i}", "source": "LemGendizedCodeFormer-manifold",
                 "split": "val" if i == 0 else "train"}
                for i in range(4)
            ]
            (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
            _, text = generate_readme_md(root)
        self.assertIn("| Train | 3 |", text)
        self.assertIn("| Val | 1 |", text)
        self.assertIn("- Real vs Fake Faces", text)
        self.assertIn("no per-source counts", text)
        self.assertIn("CC-BY-NC-4.0", text)
        self.assertIn("<https://arxiv.org/abs/2206.11253>", text)
        self.assertNotIn("| Source Dataset |", text)

    def test_registry_refs_map_to_public_pages(self) -> None:
        """Kaggle and Hugging Face refs become their public URLs; local refs are dropped."""
        self.assertEqual(
            ref_url("kaggle://owner/set"), "https://www.kaggle.com/datasets/owner/set"
        )
        self.assertEqual(
            ref_url("hf://org/name:file.tgz"), "https://huggingface.co/datasets/org/name"
        )
        self.assertIsNone(ref_url("local://raw-sets/X/"))

    def test_shard_folder_description(self) -> None:
        """Shards are described by their real extension, splits and members."""
        with tempfile.TemporaryDirectory() as tmp:
            shards = Path(tmp) / "shards"
            _write_shard(shards / "train" / "shard-00000.data", ["a.webp", "b.webp"])
            _write_shard(shards / "val" / "shard-00000.data", ["c.webp"])
            desc = _describe_directory(shards)
        self.assertIn("2 tar-format .data files", desc)
        self.assertIn("the train, val splits", desc)
        self.assertIn("one .webp file per sample", desc)
        self.assertNotIn("image-tensor", desc)


class NoLogSuppressionTests(unittest.TestCase):
    """K-DS-8: the kagglehub logger keeps its level; version banners are not silenced."""

    def test_kagglehub_logger_not_raised(self) -> None:
        """Importing the Kaggle CLI module leaves the kagglehub logger level alone."""
        self.assertNotEqual(logging.getLogger("kagglehub").level, logging.ERROR)


if __name__ == "__main__":
    unittest.main()
