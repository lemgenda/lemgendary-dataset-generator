"""Typed option groups for the Typer commands.

Commands with many CLI options declare them once as fields of a frozen
dataclass (``Annotated[..., typer.Option(...)]`` per field) and receive a
single instance instead of a long parameter list. :func:`options_command`
flattens the dataclass (and any nested option-group dataclasses, in field
order) into the keyword-only signature Typer inspects, so the generated
``--help`` and option parsing are identical to a hand-written signature.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer

from commands.common import _resolve_target

__all__ = [
    "DeviceOpt",
    "DryRunOpt",
    "ManifoldOpt",
    "ModelOpt",
    "NoServerOpt",
    "SampleOpt",
    "TargetOptions",
    "flat_values",
    "options_command",
    "require_target",
]

OptionsT = TypeVar("OptionsT")

# ─── shared option aliases ──────────────────────────────────────────────────
ModelOpt = Annotated[str | None, typer.Option("--model", "-m", help="Registry key")]
ManifoldOpt = Annotated[
    str | None, typer.Option("--manifold", help="Direct path to a manifold folder")
]
DeviceOpt = Annotated[str, typer.Option("--device", help="cpu | cuda | cuda:N")]
SampleOpt = Annotated[int | None, typer.Option("--sample", help="Cap the image scan")]
DryRunOpt = Annotated[bool, typer.Option("--dry-run")]
NoServerOpt = Annotated[
    bool, typer.Option("--no-server", help="Bypass API server and execute in-process")
]


@dataclasses.dataclass(frozen=True, kw_only=True)
class TargetOptions:
    """``--model`` / ``--manifold`` pair that selects an existing manifold."""

    model: ModelOpt = None
    manifold: ManifoldOpt = None


def require_target(target: TargetOptions) -> Path:
    """Resolve the selected manifold, or exit with code 1 when it is missing."""
    path = _resolve_target(target.model, target.manifold)
    if path is None:
        raise typer.Exit(code=1)
    return path


# ─── dataclass -> Typer signature ───────────────────────────────────────────
def _leaf_parameters(options_cls: type) -> list[inspect.Parameter]:
    """Keyword-only parameters for every leaf field of an option-group dataclass."""
    hints = typing.get_type_hints(options_cls, include_extras=True)
    params: list[inspect.Parameter] = []
    for field in dataclasses.fields(options_cls):
        hint = hints[field.name]
        if dataclasses.is_dataclass(hint):
            params.extend(_leaf_parameters(typing.cast(type, hint)))
            continue
        default = inspect.Parameter.empty
        if field.default is not dataclasses.MISSING:
            default = field.default
        params.append(
            inspect.Parameter(
                field.name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=hint
            )
        )
    return params


def _build_options(options_cls: type[OptionsT], values: dict[str, Any]) -> OptionsT:
    """Instantiate an option-group dataclass (and nested groups) from flat values."""
    hints = typing.get_type_hints(options_cls)
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(typing.cast(Any, options_cls)):
        hint = hints[field.name]
        if dataclasses.is_dataclass(hint):
            kwargs[field.name] = _build_options(typing.cast(type, hint), values)
        elif field.name in values:
            kwargs[field.name] = values[field.name]
    return options_cls(**kwargs)


def flat_values(options: Any) -> dict[str, Any]:
    """Leaf field values of an option-group instance (nested groups inlined), in order."""
    values: dict[str, Any] = {}
    for field in dataclasses.fields(options):
        value = getattr(options, field.name)
        if dataclasses.is_dataclass(value):
            values.update(flat_values(value))
        else:
            values[field.name] = value
    return values


def options_command(
    options_cls: type[OptionsT],
) -> Callable[[Callable[[OptionsT], None]], Callable[..., None]]:
    """Expose ``options_cls`` fields as CLI options of the decorated command body.

    The returned callable keeps the body's name and docstring (Typer uses them
    for the command name and help) and accepts the flattened options as
    keyword arguments.
    """
    params = _leaf_parameters(options_cls)
    names = [param.name for param in params]
    if len(names) != len(set(names)):
        raise TypeError(f"Duplicate option field names in {options_cls.__name__}: {names}")

    def decorate(body: Callable[[OptionsT], None]) -> Callable[..., None]:
        @functools.wraps(body)
        def command(**values: Any) -> None:
            body(_build_options(options_cls, values))

        command.__dict__["__signature__"] = inspect.Signature(params, return_annotation=None)
        command.__annotations__ = {param.name: param.annotation for param in params}
        command.__annotations__["return"] = None
        return command

    return decorate
