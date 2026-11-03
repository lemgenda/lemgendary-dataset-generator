"""Selection and confirmation of the legacy manifolds to modernize."""

from __future__ import annotations

import argparse


# ─── Display ────────────────────────────────────────────────────────────────
def _print_eligible(eligible: list[dict]) -> None:
    print()
    print("── Eligible Manifolds for Modernization ────────────────────────────")
    print(f"{'#':<4} {'Source Manifold (Legacy)':<48} {'Target Modern Manifold':<48}")
    print(f"{'─'*4} {'─'*48} {'─'*48}")
    for i, item in enumerate(eligible, 1):
        status = " [RESUMABLE]" if item.get("is_resume") else ""
        print(f"{i:<4} {item['folder_name']:<48} {item['target_name'] + status:<48}")
    print()


def _selection_index(part: str, count: int) -> int | None:
    """Return the 0-based index named by a 1-based ``part``, or None if unusable."""
    try:
        idx = int(part) - 1
    except ValueError:
        print(f"[WARN] Ignoring invalid selection: '{part}'")
        return None
    return idx if 0 <= idx < count else None


def _parse_selection(raw: str, eligible: list[dict]) -> list[dict]:
    raw = raw.strip().lower()
    if raw in ("a", "all"):
        return list(eligible)
    parts = [part.strip() for part in raw.split(",")]
    indices = [_selection_index(part, len(eligible)) for part in parts if part]
    return [eligible[idx] for idx in indices if idx is not None]


def _match_query(item: dict, query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False
    return q in {
        item["folder_name"].lower(),
        item["target_name"].lower(),
        item["base_name"].lower(),
        (item["kaggle_key"] or "").lower(),
    }


def _select_from_datasets_arg(
    raw: str, eligible: list[dict]
) -> list[dict]:
    queries = [q.strip() for q in raw.split(",") if q.strip()]
    picked: list[dict] = []
    matched_queries: set[str] = set()
    for item in eligible:
        for q in queries:
            if _match_query(item, q):
                picked.append(item)
                matched_queries.add(q.lower())
                break
    unmatched = [q for q in queries if q.lower() not in matched_queries]
    for q in unmatched:
        print(f"[WARN] No eligible manifold matched: '{q}'")
    return picked


def select_manifolds(args: argparse.Namespace, eligible: list[dict]) -> tuple[list[dict], int]:
    """Pick manifolds from ``--datasets``, ``--all`` or an interactive prompt.

    Returns (selected, exit_code); an empty selection carries the exit code
    the run must end with (0 when the user cancelled, 1 when nothing matched).
    """
    if args.datasets:
        selected = _select_from_datasets_arg(args.datasets, eligible)
    elif args.all:
        selected = list(eligible)
    else:
        _print_eligible(eligible)
        try:
            raw = input("Select (comma-separated numbers, or 'a' for all): ").strip()
        except KeyboardInterrupt:
            print("\n[ABORTED] Cancelled by user.")
            return [], 0
        selected = _parse_selection(raw, eligible)
    if not selected:
        print("[ERROR] No manifolds selected.")
        return [], 1
    return selected, 0


# ─── Confirmation ───────────────────────────────────────────────────────────
def _confirm(selected: list[dict], override_format: str | None = None) -> bool:
    print()
    print("── Modernization Plan ───────────────────────────────────────────────")
    for item in selected:
        status = " (RESUME partial)" if item.get("is_resume") else " (new folder)"
        fmt_disp = (override_format or item.get("canonical_format", "directory")).upper()
        print(
            f"  {item['folder_name']}  ->  {item['target_name']}{status}  [Canonical: {fmt_disp}]"
        )
    print()
    print(f"  Total: {len(selected)} modernized manifold(s) will be processed.")
    print("  Zero Duplication: Intermediate loose files will be purged for containerized sets.")
    print("  Legacy manifold folder(s) will be preserved untouched.")
    print()
    try:
        ans = input("Type YES to confirm: ").strip()
    except KeyboardInterrupt:
        print("\n[ABORTED] Cancelled by user.")
        return False
    return ans == "YES"
