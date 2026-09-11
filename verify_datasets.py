#!/usr/bin/env python3
"""
Pre-commit Verification Suite for LemGendary Datasets Hub.

Enforces four mandatory gates:
1. Python syntax and bytecode compilation (python -m py_compile)
2. Static type safety verification (pyright)
3. Code quality and linting verification (pylint)
4. Markdown documentation linting (markdownlint-cli)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parent

# Resolve python interpreter within virtual environment
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)


def log_header(title: str):
    print(f"\n{'=' * 70}")
    print(f" [GATE] {title}")
    print(f"{'=' * 70}")


def get_staged_files() -> list[Path]:
    try:
        res = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        files = []
        for line in res.stdout.splitlines():
            line = line.strip()
            if line:
                p = REPO_ROOT / line
                if p.exists():
                    files.append(p)
        return files
    except Exception:
        return []


def check_python_compilation(target_files: list[Path] | None = None) -> bool:
    log_header("GATE 1: Python Syntax & Compilation (python -m py_compile)")
    if target_files:
        py_files = [p for p in target_files if p.suffix.lower() == ".py"]
    else:
        py_files = sorted(list(REPO_ROOT.glob("*.py")))

    if not py_files:
        print("[INFO] No Python files to compile.")
        return True

    print(f"[RUN] Compiling {len(py_files)} Python source files...")
    has_errors = False
    for py_file in py_files:
        cmd = [str(VENV_PYTHON), "-m", "py_compile", str(py_file)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[FAIL] Syntax error in {py_file.name}:")
            if res.stderr:
                print(res.stderr.strip())
            has_errors = True
        else:
            print(f" [OK] {py_file.name}")

    if has_errors:
        print("[FAIL] One or more Python files failed syntax compilation.")
        return False

    print(f"[PASS] All {len(py_files)} Python files compiled cleanly.")
    return True


def check_pyright(target_files: list[Path] | None = None) -> bool:
    log_header("GATE 2: Static Type Safety (pyright)")
    if target_files:
        py_files = [p for p in target_files if p.suffix.lower() == ".py"]
        if not py_files:
            print("[INFO] No Python files to type-check.")
            return True
        cmd = [str(VENV_PYTHON), "-m", "pyright"] + [str(p) for p in py_files]
    else:
        cmd = [str(VENV_PYTHON), "-m", "pyright", "."]

    print("[RUN] Running pyright type checking...")
    try:
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if res.stdout:
            print(res.stdout.strip())
        if res.stderr:
            print(res.stderr.strip())
        if res.returncode == 0:
            print("[PASS] Pyright type checking passed with 0 errors.")
            return True
        print("[FAIL] Pyright reported type violations.")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run pyright: {e}")
        return False


def check_pylint(target_files: list[Path] | None = None) -> bool:
    log_header("GATE 3: Code Quality & Linting (pylint)")
    if target_files:
        py_files = [p for p in target_files if p.suffix.lower() == ".py"]
    else:
        py_files = sorted(list(REPO_ROOT.glob("*.py")))

    if not py_files:
        print("[INFO] No Python files to lint.")
        return True

    cmd = [str(VENV_PYTHON), "-m", "pylint"] + [str(p) for p in py_files]
    print(f"[RUN] Running pylint on {len(py_files)} Python files...")
    try:
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if res.stdout:
            print(res.stdout.strip())
        if res.stderr:
            print(res.stderr.strip())
        if res.returncode == 0:
            print("[PASS] Pylint code quality check passed.")
            return True
        print("[FAIL] Pylint reported linting violations.")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run pylint: {e}")
        return False


def check_markdown_lint(target_files: list[Path] | None = None) -> bool:
    log_header("GATE 4: Markdown Documentation Linting (markdownlint-cli)")
    if target_files:
        md_files = [p for p in target_files if p.suffix.lower() == ".md"]
    else:
        md_files = sorted(
            list(REPO_ROOT.glob("*.md")) +
            list((PROJECT_ROOT / "LemGendaryDatasets").glob("*/README.md"))
        )

    if not md_files:
        print("[INFO] No Markdown files to lint.")
        return True

    cfg_path = PROJECT_ROOT / ".markdownlint.yaml"
    cmd = ["npx.cmd" if os.name == "nt" else "npx", "markdownlint-cli"]
    if cfg_path.exists():
        cmd.extend(["-c", str(cfg_path)])
    cmd.extend([str(p) for p in md_files])

    print(f"[RUN] Linting {len(md_files)} Markdown files...")
    try:
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[PASS] All {len(md_files)} Markdown files passed with 0 errors/warnings.")
            return True
        print("[FAIL] Markdown lint violations detected:")
        if res.stdout:
            print(res.stdout.strip())
        if res.stderr:
            print(res.stderr.strip())
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run markdownlint-cli: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Pre-commit Verification Suite for LemGendary Datasets")
    parser.add_argument("--staged", action="store_true", help="Only validate git-staged files")
    args = parser.parse_args()

    print("=" * 70)
    print(" LEMGENDARY DATASETS - PRE-COMMIT AUDIT SUITE")
    print("=" * 70)

    target_files = None
    if args.staged:
        staged = get_staged_files()
        if staged:
            target_files = staged
            print(f"[INFO] Running in --staged mode for {len(staged)} staged files.")
        else:
            print("[INFO] No staged files found. Skipping check.")
            sys.exit(0)

    g1 = check_python_compilation(target_files)
    g2 = check_pyright(target_files)
    g3 = check_pylint(target_files)
    g4 = check_markdown_lint(target_files)

    log_header("DATASETS AUDIT SUMMARY")
    print(f"  Gate 1: Python Syntax & Compilation (py_compile) : {'PASSED' if g1 else 'FAILED'}")
    print(f"  Gate 2: Static Type Safety (pyright)             : {'PASSED' if g2 else 'FAILED'}")
    print(f"  Gate 3: Code Quality Linting (pylint)            : {'PASSED' if g3 else 'FAILED'}")
    print(f"  Gate 4: Markdown Linting (markdownlint)          : {'PASSED' if g4 else 'FAILED'}")
    print("=" * 70)

    if g1 and g2 and g3 and g4:
        print("[SUCCESS] All pre-commit datasets checks PASSED successfully.\n")
        sys.exit(0)
    else:
        print("[ABORT] One or more pre-commit datasets checks FAILED.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
