"""Export the frozen ``openapi.json`` contract from the live FastAPI app.

Usage (from the project root)::

    python -m tools.export_openapi          # rewrite openapi.json
    python -m tools.export_openapi --check  # exit 1 if openapi.json drifted

The output is deterministic: the app's own key order, 2-space indent, UTF-8,
LF line endings and a trailing newline. ``tests/test_openapi_contract.py``
fails when the committed file differs from this rendering.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from api.server import create_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPENAPI_FILE = PROJECT_ROOT / "openapi.json"


def render_openapi(schema: dict[str, Any]) -> str:
    """Serialise an OpenAPI schema dict to the canonical committed text."""
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def build_schema() -> dict[str, Any]:
    """Build the OpenAPI schema from the live app object."""
    return create_app().openapi()


def main(argv: list[str] | None = None) -> int:
    """Write openapi.json, or with ``--check`` exit 1 when it is out of date."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="Exit 1 if openapi.json differs from the app."
    )
    args = parser.parse_args(argv)

    rendered = render_openapi(build_schema())
    if args.check:
        current = OPENAPI_FILE.read_text(encoding="utf-8") if OPENAPI_FILE.exists() else ""
        if current != rendered:
            print(
                "openapi.json is out of date; run: python -m tools.export_openapi", file=sys.stderr
            )
            return 1
        print("openapi.json is up to date.")
        return 0

    OPENAPI_FILE.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {OPENAPI_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
