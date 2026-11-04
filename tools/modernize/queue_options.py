"""Option objects of a modernization queue run."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class CleanupOptions:
    """Which source material a successful conversion deletes."""

    delete_zip: bool = True
    delete_legacy_dir: bool = True
    purge_loose: bool = False


@dataclass(frozen=True)
class ConvertOptions:
    """How a queue item is converted into its container format."""

    shard_size: int = 5000
    workers: int | None = None
    mds_shard_size_mb: int = 512
    cleanup: CleanupOptions = field(default_factory=CleanupOptions)

    @property
    def mds_shard_size_bytes(self) -> int:
        """MDS shard size limit in bytes."""
        return self.mds_shard_size_mb * 1024 * 1024


@dataclass(frozen=True)
class QueueOptions:
    """Selection, conversion and publication options of a queue run."""

    priority_keys: list[str] | None = None
    filter_keys: list[str] | None = None
    push_kaggle: bool = False
    no_wait: bool = False
    dry_run: bool = False
    force: bool = False
    convert: ConvertOptions = field(default_factory=ConvertOptions)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> QueueOptions:
        """Build options from the flat keyword arguments of :func:`run_queue`.

        Accepts every field of this class, :class:`ConvertOptions` and
        :class:`CleanupOptions` by name; an unknown name raises TypeError.
        """
        cleanup = CleanupOptions(**_take_fields(CleanupOptions, kwargs))
        convert = ConvertOptions(cleanup=cleanup, **_take_fields(ConvertOptions, kwargs))
        return cls(convert=convert, **kwargs)

    @property
    def cleanup(self) -> CleanupOptions:
        """Source cleanup options of the conversion step."""
        return self.convert.cleanup


def _take_fields(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pop and return the entries of ``kwargs`` named like a field of dataclass ``cls``."""
    names = {f.name for f in fields(cls)}
    return {name: kwargs.pop(name) for name in list(kwargs) if name in names}


def resolve_queue_options(
    options: QueueOptions | None, kwargs: dict[str, Any]
) -> QueueOptions:
    """Return ``options``, or options built from flat ``kwargs`` (never both)."""
    if options is None:
        return QueueOptions.from_kwargs(**kwargs)
    if kwargs:
        raise TypeError(f"pass either options or keyword arguments, not both: {sorted(kwargs)}")
    return options


@dataclass(frozen=True)
class QueueArgs:
    """CLI options of ``tools.modernization_queue`` set by the API route and the CLI."""

    workers: int | None = None
    mds_shard_mb: int | None = None
    purge_source: bool = False
    push_kaggle: bool = False
    dry_run: bool = False
    force: bool = False

    def to_argv(self) -> list[str]:
        """Return the command-line arguments (without ``--datasets``)."""
        args: list[str] = []
        if self.workers:
            args.extend(["--workers", str(self.workers)])
        if self.mds_shard_mb:
            args.extend(["--mds-shard-mb", str(self.mds_shard_mb)])
        if self.purge_source:
            args.append("--purge-loose")
        else:
            args.extend(["--keep-zip", "--keep-legacy-dir"])
        switches = (
            (self.push_kaggle, "--push-kaggle"),
            (self.force, "--force"),
            (self.dry_run, "--dry-run"),
        )
        args.extend(switch for enabled, switch in switches if enabled)
        return args
