# Code Quality Policy

Applies to `lemgendary-datasets/`. Enforced by `lem-env validate --project lemgendary-datasets`.

## Suppression Policy

`../ROBOTS.md` governs; this file only restates it for this project.

1. **No inline suppressions.** No `# type: ignore` (with or without a bracket code),
   `# noqa`, `# pylint: disable` or `# pragma: no cover`. Fix the root cause instead:
   - Add return annotations, narrow unions, use `setattr` for dynamic attributes.
   - Extract a narrowing helper (`_require_text`, `_require_ann_path`).
   - Install a `*-stubs` package for third-party gaps.
   - Isolate untyped third-party dependencies into a bridge module.

2. **No configuration-based suppressions.** `.pylintrc` must not disable checks to
   hide findings. Any check still listed there is a known backlog item, tracked in the
   README changelog, and is removed as soon as its findings are fixed.

3. **No swallowed exceptions.** Never `except: pass`, bare `except:` or a silent
   `except Exception`. Catch the specific types the `try` body can raise and log
   (`logger.debug` for expected conditions, `logger.warning`/`logger.error` otherwise),
   re-raise with `raise ... from exc`, or return a typed error. Shared tuples of
   expected failure types live in one place each: `core/error_types.py`,
   `core/kaggle_errors.py` and `services/_errors.py`.

## Package Structure

- Same-package imports never need suppressions. If they do, the package's `__init__.py`
  is missing or broken — fix the package.
- Untyped third-party modules get one bridge (e.g. `mt5_bridge.py`). Consumers import
  from the bridge, not from the underlying library.

## Dynamic Attributes

For PIL, torch, and similar modules where Pyrefly can't prove an attribute exists:

```python
# Correct — no suppression needed
setattr(module, "ATTRIBUTE", value)

# Incorrect — requires a suppression
module.ATTRIBUTE = value  # type: ignore[attr-defined]
```
