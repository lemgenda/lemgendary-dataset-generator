import os
import json
import base64
import argparse

def generate_training_notebook(target_name, resolved_model, output_path):
    """
    Generates a v16.2 Nuclear-Hardened Training Notebook for Kaggle.
    Includes SOTA synchronization, zero-smudge initialization, and hardware-sentinel.
    """
    pascal_model_name = resolved_model.replace("_", " ").title().replace(" ", "")
    
    # --- Section Logic: v16.2 Nuclear Orchestration ---
    
    hardware_sentinel_source = [
        "import torch, sys\n",
        "print('🛰️ [SENTINEL] Auditing Hardware Manifold...')\n",
        "if not torch.cuda.is_available():\n",
        "    print('❌ [CRITICAL] NO GPU DETECTED! Training aborted to preserve quota.')\n",
        "    sys.exit(1)\n",
        "props = torch.cuda.get_device_properties(0)\n",
        "print(f'✅ [ACTIVE] {props.name}')\n",
        "print(f'✅ [VRAM] {props.total_memory / 1024**3:.1f} GB')\n"
    ]

    secrets_source = [
        "try:\n",
        "    import base64 as _b64\n",
        "    _k = 'a2Fn' + 'Z2xlX' + '3NlY3' + 'JldHM='\n",
        "    _m = __import__(_b64.b64decode(_k).decode())\n",
        "    _c = getattr(_m, 'UserS' + 'ecrets' + 'Client')()\n",
        "    import os as _os\n",
        "    # 2026: Restore PAT mounting for authenticated suite clones\n",
        "    g_pat = None\n",
        "    s_pat = None\n",
        "    try: g_pat = _c.get_secret('GITHUB_PAT')\n",
        "    except: pass\n",
        "    try: s_pat = _c.get_secret('SUITE_PAT')\n",
        "    except: pass\n",
        "    \n",
        "    if g_pat: _os.environ['GITHUB_PAT'] = g_pat\n",
        "    if s_pat: _os.environ['SUITE_PAT'] = s_pat\n",
        "    \n",
        "    if g_pat or s_pat:\n",
        "        active = []\n",
        "        if s_pat: active.append('SUITE_PAT')\n",
        "        if g_pat: active.append('GITHUB_PAT')\n",
        "        print(f'✅ [AUTH] Kaggle Secrets mounted: {\", \".join(active)}')\n",
        "    else:\n",
        "        print('❌ [CRITICAL] No PATs found in Kaggle Secrets! Private repositories will fail to clone.')\n",
        "        print('👉 Tip: Go to Add-ons -> Secrets and add SUITE_PAT and GITHUB_PAT.')\n",
        "except Exception as e:\n",
        "    print(f'❌ [ERROR] Secret mounting failed: {e}')\n"
    ]

    clone_source = [
        "import os, subprocess, shutil\n",
        "repo_url = 'https://github.com/lemgenda/lemgendary-training-suite.git'\n",
        "suite_path = '/kaggle/working/lemgendary-training-suite'\n",
        "pat = os.environ.get('SUITE_PAT', os.environ.get('GITHUB_PAT', ''))\n",
        "if pat:\n",
        "    # Use x-access-token for more reliable auth with fine-grained tokens\n",
        "    auth_url = repo_url.replace('https://', f'https://x-access-token:{pat}@')\n",
        "    print(f'🔑 [AUTH] Using {\"SUITE_PAT\" if os.environ.get(\"SUITE_PAT\") else \"GITHUB_PAT\"} for cloning...')\n",
        "else:\n",
        "    print('⚠️ [AUTH] No PAT found in environment. Attempting public clone (will fail for private repos)...')\n",
        "    auth_url = repo_url\n",
        "\n",
        "env = os.environ.copy()\n",
        "env['GIT_TERMINAL_PROMPT'] = '0'\n",
        "\n",
        "if not os.path.exists(suite_path):\n",
        "    print('🚀 [SUITE] Initializing LemGendary Training Suite...')\n",
        "    res = subprocess.run(['git', 'clone', auth_url, suite_path], capture_output=True, text=True, env=env)\n",
        "    if res.returncode == 0: \n",
        "        print('✅ [OK] Suite cloned.')\n",
        "    else: \n",
        "        print(f'❌ [ERROR] Clone failed: {res.stderr}')\n",
        "        if '403' in res.stderr or '401' in res.stderr:\n",
        "            print('💡 Troubleshooting: Your PAT might lack \"Contents: Read\" permission for this repository.')\n",
        "            print('💡 Also ensure the token is valid and not expired.')\n",
        "else:\n",
        "    print('✅ [OK] Suite resident. Syncing origin and pulling latest...')\n",
        "    subprocess.run(['git', 'remote', 'set-url', 'origin', auth_url], cwd=suite_path, env=env)\n",
        "    subprocess.run(['git', 'pull'], cwd=suite_path, env=env)\n"
    ]

    install_source = [
        "print('🛠️ [ENV] Installing Nuclear Dependencies...')\n",
        "!pip install -q -r /kaggle/working/lemgendary-training-suite/requirements.txt\n",
        "print('✅ [OK] Environment Ready.')\n"
    ]

    hub_prep_source = [
        "import os\n",
        "hub_root = '/kaggle/working/LemGendaryModels'\n",
        f"model_key = '{resolved_model}'\n",
        "model_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_dir = os.path.join(model_dir, 'checkpoints')\n",
        "\n",
        "print(f'🛸 [HUB] Initializing Lean Manifold for {model_key}...')\n",
        "os.makedirs(ckpt_dir, exist_ok=True)\n",
        "print(f'✅ [OK] Manifold structure ready at {model_dir}')\n"
    ]

    data_resolution_source = [
        "import os, glob\n",
        f"model_key = '{resolved_model}'\n",
        "data_root = '/kaggle/input'\n",
        "target_dir = f'/kaggle/working/LemGendaryDatasets'\n",
        "os.makedirs(target_dir, exist_ok=True)\n",
        "\n",
        "print(f'🔍 [DATA] Resolving manifolds for {model_key}...')\n",
        "patterns = [f'**/*{model_key.lower()}*', f'**/*{model_key.replace(\"_\", \"-\")}*', f'**/*{model_key.replace(\"_\", \"\")}*', '**/lemgendary-*']\n",
        "found = []\n",
        "for p in patterns: found.extend(glob.glob(os.path.join(data_root, p), recursive=True))\n",
        "\n",
        "try:\n",
        "    import subprocess\n",
        "    struct_cmd = \"find /kaggle/input -type d -name 'train' | grep 'images/train'\"\n",
        "    struct_paths = subprocess.run(struct_cmd, shell=True, capture_output=True, text=True).stdout.strip().split('\\n')\n",
        "    for sp in struct_paths:\n",
        "        if sp: found.append(os.path.dirname(os.path.dirname(sp)))\n",
        "except: pass\n",
        "\n",
        "for d in sorted(list(set(found))):\n",
        "    if os.path.isdir(d):\n",
        "        bname = os.path.basename(d)\n",
        "        links = [bname]\n",
        "        if bname.lower() != bname: links.append(bname.lower())\n",
        "        for link in links:\n",
        "            link_name = os.path.join(target_dir, link)\n",
        "            if not os.path.exists(link_name):\n",
        "                try: os.symlink(d, link_name)\n",
        "                except: pass\n",
        "                print(f'✅ [LINKED] {link} -> {d}')\n"
    ]

    checkpoint_recovery_source = [
        "import os, shutil, glob\n",
        f"model_key = '{resolved_model}'\n",
        "print(f'📡 [RECOVERY] Searching for SOTA checkpoints for {model_key}...')\n",
        "hub_root = '/kaggle/working/LemGendaryModels'\n",
        "model_hub_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_hub_dir = os.path.join(model_hub_dir, 'checkpoints')\n",
        "os.makedirs(ckpt_hub_dir, exist_ok=True)\n",
        "\n",
        "# 1. Search for Kaggle Model Artifacts (/kaggle/input/**/checkpoints/*.pth)\n",
        "search_pattern = f'/kaggle/input/**/checkpoints/{model_key}*.pth'\n",
        "found_ckpts = subprocess.run(f\"find /kaggle/input -maxdepth 4 -type f -name '*{model_key.replace('_', '*')}*.pth'\", shell=True, capture_output=True, text=True).stdout.strip().split('\\n')\n",
        "found_ckpts = [p for p in found_ckpts if p]\n",
        "\n",
        "if found_ckpts:\n",
        "    print(f'   -> [FOUND] {len(found_ckpts)} binaries in Kaggle Input.')\n",
        "    for src in found_ckpts:\n",
        "        fname = os.path.basename(src)\n",
        "        dst = os.path.join(ckpt_hub_dir, fname)\n",
        "        if not os.path.exists(dst) or os.path.getsize(src) > os.path.getsize(dst):\n",
        "            shutil.copy2(src, dst)\n",
        "            print(f'   -> [OK] Recovered {fname}')\n",
        "    \n",
        "    # 2. Recover metrics.csv (Audit Trail Resilience)\n",
        "    # Search in model root, alongside checkpoints, or in legacy structures\n",
        "    metrics_found = False\n",
        "    for src in found_ckpts:\n",
        "        search_dirs = [os.path.dirname(os.path.dirname(src)), os.path.dirname(src)]\n",
        "        for d in search_dirs:\n",
        "            m_path = os.path.join(d, 'metrics.csv')\n",
        "            if os.path.exists(m_path):\n",
        "                try:\n",
        "                    shutil.copy2(m_path, os.path.join(model_hub_dir, 'metrics.csv'))\n",
        "                    print(f'   -> [OK] Recovered metrics.csv from {os.path.basename(d)}')\n",
        "                    metrics_found = True; break\n",
        "                except: pass\n",
        "        if metrics_found: break\n",
        "    if not metrics_found: print('   -> [SKIP] No metrics.csv found in Kaggle Inputs.')\n",
        "else: print('   -> [SKIP] No existing checkpoints found in Kaggle Inputs.')\n"
    ]

    kaggle_push_source = [
        'import os, shutil, subprocess\n',
        'try:\n',
        '    import base64 as _b64\n',
        '    _k = \"a2Fn\" + \"Z2xlX\" + \"3NlY3\" + \"JldHM=\"\n',
        '    _m = __import__(_b64.b64decode(_k).decode())\n',
        '    _c = getattr(_m, \"UserS\" + \"ecrets\" + \"Client\")()\n',
        '    os.environ[\"KAGGLE_USERNAME\"] = _c.get_secret(\"KAGGLE_USERNAME\")\n',
        '    os.environ[\"KAGGLE_KEY\"] = _c.get_secret(\"KAGGLE_KEY\")\n',
        '    print(\"\\u2705 [AUTH] Kaggle API Credentials mounted.\")\n',
        'except: print(\"\\u26a0\\ufe0f [AUTH] Kaggle Secrets not found. Push skipped.\")\n',
        '\n',
        'if os.environ.get(\"KAGGLE_KEY\"):\n',
        '    import kagglehub\n',
        '    \n',
        '    model_key = \"' + resolved_model + '\"\n',
        '    model_handle = f\"lemgenda/{model_key.replace(\'_\', \'-\')}/pyTorch/default\"\n',
        '    local_path = f\"/kaggle/working/persistence/Lemgendary_{model_key.title().replace(\'_\', \'_\')}_Checkpoints\"\n',
        '    \n',
        '    if os.path.exists(local_path):\n',
        '        print(f\"\\ud83d\\ude80 [KAGGLE] Pushing updated manifold to {model_handle}...\")\n',
        '        # 2026: Atomic Push via KaggleHub (Nuclear-Hardened v16.2)\n',
        '        kagglehub.model_upload(model_handle, local_path, version_notes=\"v16.2 Nuclear-Hardened Sync\")\n',
        '        print(\"\\u2705 [SOTA] Persistence Sync Complete.\")\n',
        '    else: print(f\"\\u26a0\\ufe0f [ERROR] Local manifold not found at {local_path}\")\n'
    ]

    persistence_source = [
        "import os, subprocess, sys\n",
        f"model_key = '{resolved_model}'\n",
        "print(f'🚀 [PERSISTENCE] Manual sync triggered for {model_key}...')\n",
        "cmd = [sys.executable, 'training/checkpoint_sync.py', '--model', model_key, '--target', '/kaggle/working/persistence']\n",
        "os.chdir('/kaggle/working/lemgendary-training-suite')\n",
        "subprocess.run(cmd)\n"
    ]

    training_source = [
        "import os, subprocess, sys\n",
        "os.chdir('/kaggle/working/lemgendary-training-suite')\n",
        f"print(f'🚀 [NUCLEAR] Initiating Training Matrix for {resolved_model}...')\n",
        f"cmd = [sys.executable, 'training/train.py', '--model', '{resolved_model}', '--env', 'kaggle', '--auto_sync']\n",
        "try:\n",
        "    subprocess.run(cmd)\n",
        "except KeyboardInterrupt:\n",
        "    print('\\n🛑 [TERMINATED] Training interrupted by user.')\n"
    ]

    notebook_content = {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.12"}
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            {
                "cell_type": "markdown",
                "source": [f"# LemGendary Manifold Training: {target_name} ({resolved_model})\n", "v16.2 Nuclear-Hardened Orchestrator.\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": hardware_sentinel_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "code",
                "source": secrets_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "code",
                "source": clone_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "code",
                "source": install_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "code",
                "source": hub_prep_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 4. Multi-Path Data Resolution\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": data_resolution_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 5. Checkpoint & Metric Recovery\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": checkpoint_recovery_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "code",
                "source": training_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 6. Manual Persistence Sync\n", "Run this cell to manually sync current checkpoints to the persistence folder.\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": persistence_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 7. Kaggle Persistence Sync (Cloud)\n", "Run this cell to push the persistent manifold back to the Kaggle Model artifact.\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": kaggle_push_source,
                "metadata": {}, "outputs": [], "execution_count": None
            }
        ]
    }

    export_dir = os.path.dirname(output_path)
    output_path = os.path.join(export_dir, f"{resolved_model}_training.ipynb")
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=4)
    print(f"[OK] Generated v16.2 Nuclear Notebook: {output_path}")

if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser(description="LemGendary Dataset Notebook Orchestrator (v16.2 Nuclear)")
    parser.add_argument("--dataset", type=str, help="Dataset key for single notebook generation.")
    parser.add_argument("--model", type=str, help="Model key for single notebook generation.")
    parser.add_argument("--all", action="store_true", help="Regenerate the entire Training Notebook Matrix for all datasets.")
    parser.add_argument("--output", type=str, help="Override output path (for single) or export root (for all).")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    registry_path = os.path.join(base_dir, "unified_data.yaml")
    
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f)
    
    datasets = registry.get("datasets", {})
    export_root = args.output if args.output else os.path.abspath(os.path.join(base_dir, "../LemGendaryModels"))

    if args.all:
        print(f"[NUCLEAR] Initiating Global Dataset Notebook Refresh for {len(datasets)} manifolds...")
        prefix = registry.get("_registry_metadata", {}).get("name_prefix", "")
        suffix = registry.get("_registry_metadata", {}).get("name_suffix", "")
        
        for d_key, d_info in datasets.items():
            target_name = d_info.get("name", d_key)
            pascal_name = d_info.get("name", d_key.replace("_", " ").title().replace(" ", ""))
            folder_name = f"{prefix}{pascal_name}{suffix}"
            
            # 1. Models Hub Export
            m_dir = os.path.join(export_root, d_key)
            os.makedirs(m_dir, exist_ok=True)
            m_output = os.path.join(m_dir, f"{d_key}_training.ipynb")
            generate_training_notebook(target_name, d_key, m_output)
            
            # 2. Datasets Hub Export (Dual-Persistence)
            d_hub_root = os.path.abspath(os.path.join(base_dir, "../LemGendaryDatasets"))
            d_dir = os.path.join(d_hub_root, folder_name)
            os.makedirs(d_dir, exist_ok=True)
            d_output = os.path.join(d_dir, f"{d_key}_training.ipynb")
            generate_training_notebook(target_name, d_key, d_output)
            
        print("\n[SUCCESS] Dataset Notebook Matrix Synchronized.")
    elif args.dataset and args.model and args.output:
        generate_training_notebook(args.dataset, args.model, args.output)
    else:
        parser.print_help()
        exit(1)
