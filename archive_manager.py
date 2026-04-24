import os
import sys
import zipfile
import argparse
from pathlib import Path
from tqdm import tqdm

def verify_zip(zip_path):
    """Check if zip is fully readable and uncorrupted."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            bad_file = zf.testzip()
            if bad_file:
                print(f"[ERROR] Corrupted file in zip: {bad_file}")
                return False
        return True
    except Exception as e:
        print(f"[ERROR] Invalid zip file {zip_path}: {e}")
        return False

def smart_extract(zip_path, dest_dir):
    """Extract only missing files from zip and delete zip on success."""
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Opening archive: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = zf.infolist()
            total_files = len(members)
            to_extract = []
            
            # Check what's missing
            for member in members:
                target_path = dest_path / member.filename
                # If it's a file and doesn't exist or is 0 bytes
                if not member.is_dir():
                    if not target_path.exists() or target_path.stat().st_size == 0:
                        to_extract.append(member)
            
            print(f"Found {len(to_extract)} missing files out of {total_files} total.")
            
            if to_extract:
                with tqdm(total=len(to_extract), desc="Extracting", unit="file", colour="green") as pbar:
                    for member in to_extract:
                        zf.extract(member, dest_path)
                        pbar.update(1)
            else:
                print("All files already extracted.")
                
        # If we reach here, extraction was 100% successful or already done
        print(f"Extraction successful. Deleting source archive: {zip_path}")
        os.remove(zip_path)
        return True
    except Exception as e:
        print(f"[ERROR] Extraction failed for {zip_path}: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LemGendary Smart Archive Manager")
    parser.add_argument("--zip", type=str, required=True, help="Path to the zip file")
    parser.add_argument("--dest", type=str, required=True, help="Path to extract destination")
    parser.add_argument("--action", type=str, choices=["verify", "extract"], required=True, help="Action to perform")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.zip):
        print(f"[ERROR] Zip file not found: {args.zip}")
        sys.exit(1)
        
    if args.action == "verify":
        success = verify_zip(args.zip)
        sys.exit(0 if success else 1)
    elif args.action == "extract":
        success = smart_extract(args.zip, args.dest)
        sys.exit(0 if success else 1)
