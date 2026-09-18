#!/usr/bin/env python3
"""
LemGendary Kaggle Metadata & Column Descriptors Synchronizer CLI
================================================================
Synchronizes dataset-metadata.json, licenses, keywords, and column-level
schema descriptors directly to Kaggle's backend via the Dataset Settings API.
Can be executed standalone without re-uploading dataset archives.
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "tools" else Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.common_sync import push_kaggle_dataset_metadata


def main():
    """CLI entrypoint to push dataset metadata to Kaggle."""
    parser = argparse.ArgumentParser(description="Synchronize dataset metadata and column descriptors to Kaggle.")
    parser.add_argument("--repo_id", default="lemtreursi/lemgendizedforexuniverselarge", help="Kaggle dataset handle (owner/slug)")
    parser.add_argument("--metadata_dir", default=r"LemGendaryDatasets\LemGendizedForexUniverseLarge", help="Directory containing dataset-metadata.json")
    args = parser.parse_args()

    meta_path = Path(args.metadata_dir)
    success = push_kaggle_dataset_metadata(args.repo_id, meta_path)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
