import os
import shutil

raw_dir = "raw-sets"
deleted_files = 0
deleted_dirs = 0
freed_space = 0

print(f"[JANITOR] Starting deep cleanup of {raw_dir}...")

for root, dirs, files in os.walk(raw_dir, topdown=False):
    for name in files:
        filepath = os.path.join(root, name)
        try:
            # 1. Zero-byte files
            if os.path.getsize(filepath) == 0:
                print(f"  [DELETE] Zero-byte file: {name}")
                os.remove(filepath)
                deleted_files += 1
            # 2. Broken zip/tmp files (less than 1KB)
            elif name.endswith((".zip", ".tmp", ".part")) and os.path.getsize(filepath) < 1024:
                print(f"  [DELETE] Corrupted fragment: {name}")
                os.remove(filepath)
                deleted_files += 1
        except Exception:
            pass

    for name in dirs:
        dirpath = os.path.join(root, name)
        try:
            # 3. Empty directories
            if not os.listdir(dirpath):
                # print(f"  [DELETE] Empty folder: {name}")
                os.rmdir(dirpath)
                deleted_dirs += 1
        except Exception:
            pass

print(f"\n[CLEANUP COMPLETE]")
print(f"  - Files Purged: {deleted_files}")
print(f"  - Folders Purged: {deleted_dirs}")
