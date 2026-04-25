import os
import argparse
from huggingface_hub import snapshot_download, login
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="LemGendary HF Downloader")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Repo ID")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--repo_type", type=str, default="model", help="Repository type: model, dataset, space")
    parser.add_argument("--token_file", type=str, default=".huggingface_token", help="Path to token file")
    args = parser.parse_args()

    # Auth
    token = None
    if os.path.exists(args.token_file):
        with open(args.token_file, "r") as f:
            token = f.read().strip()
            if token:
                login(token=token)
                print(f"[AUTH] Logged into Hugging Face via {args.token_file}")

    # Download (Surgical Mode v2.0)
    print(f"[HF] Starting {args.repo_type} download for {args.repo_id}...")
    import tarfile
    import zipfile
    from huggingface_hub import list_repo_files, hf_hub_download

    # Sub-Repo Targeting (Syntax: repo_id:filename)
    target_file = None
    repo_id = args.repo_id
    if ":" in repo_id:
        repo_id, target_file = repo_id.split(":", 1)
        print(f"[SURGICAL] Targeting specific file: {target_file}")

    try:
        if target_file:
            targets = [target_file]
        else:
            # Get list of files to see if we should be surgical
            all_files = list_repo_files(repo_id=repo_id, repo_type=args.repo_type, token=token)
            human_technical_targets = ['spaq.tgz', 'live.tgz', 'csiq.tgz', 'tid2013.tgz']
            targets = [f for f in all_files if f.lower() in human_technical_targets]
            
            if not targets:
                print(f"[HF] No specific human targets found. Falling back to archives & parquets...")
                targets = [f for f in all_files if f.endswith(('.tgz', '.tar.gz', '.zip', '.parquet'))]
        
        if not targets:
            print(f"[HF] No specific targets found. Performing full snapshot download...")
            snapshot_download(
                repo_id=repo_id,
                repo_type=args.repo_type,
                local_dir=args.output_dir,
                token=token
            )
        else:
            print(f"[HF] Processing {len(targets)} files.")
            for t_file in targets:
                # Resolve output directory
                if t_file.endswith(('.tgz', '.tar.gz', '.zip')):
                    slug = t_file.replace('.tgz', '').replace('.tar.gz', '').replace('.zip', '')
                    dedicated_dir = Path(args.output_dir).parent / slug
                else:
                    dedicated_dir = Path(args.output_dir)
                
                dedicated_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"  [PULL] {t_file} -> {dedicated_dir}")
                
                # High-Visibility Download via requests + tqdm (Resumable)
                try:
                    from tqdm import tqdm
                    import requests
                    from huggingface_hub import hf_hub_url
                    
                    url = hf_hub_url(repo_id=repo_id, filename=t_file, repo_type=args.repo_type)
                    headers = {"Authorization": f"Bearer {token}"} if token else {}
                    
                    path = dedicated_dir / t_file
                    existing_size = 0
                    if path.exists():
                        existing_size = path.stat().st_size
                        headers["Range"] = f"bytes={existing_size}-"
                        print(f"  [RESUME] Found partial file ({existing_size/(1024*1024):.2f} MB). Picking up...")

                    response = requests.get(url, headers=headers, stream=True)
                    # If 206 Partial Content, we are resuming
                    total_size = int(response.headers.get('content-length', 0))
                    if response.status_code == 206:
                        total_size += existing_size
                    
                    mode = "ab" if existing_size > 0 else "wb"
                    with open(path, mode) as f, tqdm(
                        desc=f"   [DL] {t_file}",
                        total=total_size,
                        initial=existing_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        bar_format='{desc}: {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                    ) as pbar:
                        for data in response.iter_content(chunk_size=1024*1024):
                            size = f.write(data)
                            pbar.update(size)
                            
                    # Auto-Extract to dedicated_dir
                    if t_file.endswith(('.tgz', '.tar.gz')):
                        print(f"  [UNPACK] Extracting TGZ: {t_file}...")
                        import tarfile
                        with tarfile.open(path, "r:gz") as tar:
                            # Python 3.12+ requires explicit filter to prevent DeprecationWarning crashes
                            if hasattr(tarfile, 'data_filter'):
                                tar.extractall(path=dedicated_dir, filter='data')
                            else:
                                tar.extractall(path=dedicated_dir)
                        os.remove(path)
                    elif t_file.endswith('.zip'):
                        print(f"  [UNPACK] Extracting ZIP: {t_file}...")
                        import zipfile
                        with zipfile.ZipFile(path, 'r') as zip_ref:
                            zip_ref.extractall(dedicated_dir)
                        os.remove(path)
                except Exception as e:
                    print(f"  [WARN] Requests failed, falling back to basic HF-Hub: {e}")
                    from huggingface_hub import hf_hub_download
                    path = hf_hub_download(repo_id=repo_id, filename=t_file, repo_type=args.repo_type, local_dir=dedicated_dir, token=token)

        print(f"[SUCCESS] {args.repo_id} processed to {args.output_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to download {args.repo_id}: {e}")
        exit(1)

if __name__ == "__main__":
    main()
