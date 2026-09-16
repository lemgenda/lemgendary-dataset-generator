# Code Quality Policy

Applies to `lemgendary-datasets/`. Enforced by `lem-env validate --project lemgendary-datasets`.

## Suppression Policy

1. **No `# type: ignore` without a bracket code.** Always `# type: ignore[error-code]`.
   Never the bare form. The bracket code tells Pyrefly exactly which error to suppress
   and lets every other error in the same file surface.

2. **No suppression without a trailing reason.** One line, `# reason`, on the same
   line. Example:

   ```python
   import MetaTrader5 as _mt5  # type: ignore[import-untyped]  # MT5 ships no PEP 561 stubs
   ```

3. **Prefer fixes over suppressions.** Order of preference:
   - Fix the root cause (add return annotations, narrow unions, use `setattr` for
     dynamic attributes)
   - Extract a narrowing helper (`_require_text`, `_require_ann_path`)
   - Install a `*-stubs` package for third-party gaps
   - Isolate untyped third-party deps into a bridge module
   - Last resort: narrow suppression with a reason

4. **Never bare-except silently.** Always `except SpecificError: pass  # reason`.
   Broad `except Exception` is permitted only when the handler logs the exception
   or performs meaningful recovery.

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
