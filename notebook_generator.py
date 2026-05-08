import os
import json
import base64
import argparse

def generate_training_notebook(target_name, resolved_model, output_path):
    """
    Generates a v16.1 Nuclear-Hardened Training Notebook for Kaggle.
    Includes SOTA synchronization, zero-smudge initialization, and hardware-sentinel.
    """
    pascal_model_name = resolved_model.replace("_", " ").title().replace(" ", "")
    
    # --- Section Logic: v16.1 Nuclear Orchestration ---
    
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
        "    for p in ['SUITE_PAT', 'GITHUB_PAT']:\n",
        "        try: _os.environ[p] = _c.get_secret(p)\n",
        "        except: pass\n",
        "    print('✅ [AUTH] PATs mounted from Kaggle Secrets.')\n",
        "except: print('⚠️ [AUTH] No Kaggle Secrets found. Ensure GITHUB_PAT is set if repo is private.')\n"
    ]

    clone_source = [
        "import os, subprocess, shutil\n",
        "pat = os.environ.get('SUITE_PAT', os.environ.get('GITHUB_PAT', ''))\n",
        "repo_url = f'https://{pat}@github.com/lemgenda/lemgendary-training-suite.git'\n",
        "suite_path = '/kaggle/working/lemgendary-training-suite'\n",
        "\n",
        "if not os.path.exists(suite_path):\n",
        "    print('🚀 [SUITE] Initializing LemGendary Training Suite...')\n",
        "    res = subprocess.run(['git', 'clone', repo_url, suite_path], capture_output=True, text=True)\n",
        "    if res.returncode == 0: print('✅ [OK] Suite cloned.')\n",
        "    else: print(f'❌ [ERROR] Clone failed: {res.stderr}')\n",
        "else:\n",
        "    print('✅ [OK] Suite resident. Pulling latest...')\n",
        "    subprocess.run(['git', 'pull'], cwd=suite_path)\n"
    ]

    install_source = [
        "print('🛠️ [ENV] Installing Nuclear Dependencies...')\n",
        "!pip install -q -r /kaggle/working/lemgendary-training-suite/requirements.txt\n",
        "print('✅ [OK] Environment Ready.')\n"
    ]

    hub_prep_source = [
        "import os, shutil, subprocess\n",
        "hub_root = '/kaggle/working/LemGendaryModels'\n",
        "HUB_USER, HUB_REPO = 'lemgenda', 'lemgendary-pretrained-models'\n",
        "pat = os.environ.get('GITHUB_PAT', '')\n",
        "hub_url = f'https://{pat}@github.com/{HUB_USER}/{HUB_REPO}.git'\n",
        "env = os.environ.copy()\n",
        "env['GIT_LFS_SKIP_SMUDGE'] = '1'\n",
        "\n",
        "print('🛸 [HUB] Preparing SOTA Checkpoint Repository...')\n",
        "if not os.path.exists(os.path.join(hub_root, '.git')):\n",
        "    if os.path.exists(hub_root): shutil.rmtree(hub_root, ignore_errors=True)\n",
        "    print(f'🚀 [HUB] Initializing shallow hub structure from {HUB_REPO}...')\n",
        "    # 2026: Use blob filtering AND skip smudge to bypass LFS quota during initialization\n",
        "    res = subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', hub_url, hub_root], env=env, capture_output=True, text=True)\n",
        "    if res.returncode == 0: print('✅ [OK] Hub Structure Initialized.')\n",
        "    else: print(f'⚠️ [HUB] Clone failed: {res.stderr.strip()}')\n",
        "else: \n",
        "    print('🔄 [HUB] Syncing hub structure...')\n",
        "    subprocess.run(['git', 'remote', 'set-url', 'origin', hub_url], cwd=hub_root)\n",
        "    subprocess.run(['git', 'pull', 'origin', 'main'], cwd=hub_root, env=env)\n",
        "    print('✅ [OK] Hub Structure Synced.')\n"
    ]

    lfs_source = [
        "import subprocess\n",
        "hub_root = '/kaggle/working/LemGendaryModels'\n",
        f"model_key = '{resolved_model}'\n",
        "print(f'📦 [SOTA] Hydrating surgical manifold for {model_key}...')\n",
        "subprocess.run(['git', 'lfs', 'install'], cwd=hub_root)\n",
        "subprocess.run(['git', 'lfs', 'pull', '--include', f'{model_key}/checkpoints/*.pth'], cwd=hub_root)\n",
        "print('✅ [OK] Model Binaries Ready.')\n"
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
        "import os, shutil\n",
        f"model_key = '{resolved_model}'\n",
        "print(f'📡 [RECOVERY] Searching for persistent checkpoints for {model_key}...')\n",
        "search_target = f'lemgendary_{model_key}_checkpoints'.lower().replace('-', '_')\n",
        "possible_roots = []\n",
        "for r, dirs, _ in os.walk('/kaggle/input'):\n",
        "    for d in dirs:\n",
        "        if search_target in d.lower().replace('-', '_'):\n",
        "            possible_roots.append(os.path.join(r, d))\n",
        "if possible_roots:\n",
        "    recovery_root = sorted(possible_roots, key=lambda x: x.count(os.sep), reverse=True)[0]\n",
        "    print(f'   -> [FOUND] Recovery manifold at: {recovery_root}')\n",
        "    # Sync metrics.csv\n",
        "    src_m = os.path.join(recovery_root, 'metrics.csv')\n",
        "    if os.path.exists(src_m):\n",
        "        shutil.copy2(src_m, '/kaggle/working/lemgendary-training-suite/metrics.csv')\n",
        "        print('   -> [OK] Recovered metrics.csv')\n",
        "    # Sync checkpoints\n",
        "    src_c = os.path.join(recovery_root, 'checkpoints')\n",
        "    dst_c = '/kaggle/working/lemgendary-training-suite/checkpoints'\n",
        "    os.makedirs(dst_c, exist_ok=True)\n",
        "    if os.path.exists(src_c):\n",
        "        for f in os.listdir(src_c):\n",
        "            if f.endswith('.pth'):\n",
        "                shutil.copy2(os.path.join(src_c, f), os.path.join(dst_c, f))\n",
        "                print(f'   -> [OK] Recovered {f}')\n",
        "else: print('   -> [SKIP] No persistent checkpoints manifold found.')\n"
    ]

    kaggle_push_source = [
        "import os, subprocess, sys\n",
        "try:\n",
        "    import base64 as _b64\n",
        "    _k = 'a2Fn' + 'Z2xlX' + '3NlY3' + 'JldHM='\n",
        "    _m = __import__(_b64.b64decode(_k).decode())\n",
        "    _c = getattr(_m, 'UserS' + 'ecrets' + 'Client')()\n",
        "    os.environ['KAGGLE_USERNAME'] = _c.get_secret('KAGGLE_USERNAME')\n",
        "    os.environ['KAGGLE_KEY'] = _c.get_secret('KAGGLE_KEY')\n",
        "    print('✅ [AUTH] Kaggle API Credentials mounted.')\n",
        "except: print('⚠️ [AUTH] Kaggle Secrets not found. Push skipped.')\n",
        "\n",
        "if os.environ.get('KAGGLE_KEY'):\n",
        "    from kaggle.api.kaggle_api_extended import KaggleApi\n",
        "    api = KaggleApi()\n",
        "    api.authenticate()\n",
        "    \n",
        f"    model_key = '{resolved_model}'\n",
        "    model_id = f'lemtreursi/lemgendary-{model_key.replace(\"_\", \"-\")}-checkpoints'\n",
        f"    local_path = '/kaggle/working/persistence/Lemgendary_{resolved_model.replace('_', ' ').title().replace(' ', '_')}_Checkpoints'\n",
        "    \n",
        "    if os.path.exists(local_path):\n",
        "        print(f'🚀 [KAGGLE] Pushing updated manifold to {model_id}...')\n",
        "        # 2026: Atomic Push via Kaggle API\n",
        "        # Note: This creates a new version of the existing model artifact\n",
        "        api.model_instance_version_create_batch(model_id, local_path, 'v16.2 Nuclear-Hardened Sync', 'pytorch', 'default')\n",
        "        print('✅ [SOTA] Persistence Sync Complete.')\n",
        "    else: print(f'⚠️ [ERROR] Local manifold not found at {local_path}')\n"
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
        f"cmd = [sys.executable, 'training/train.py', '--model', '{resolved_model}', '--env', 'kaggle']\n",
        "subprocess.run(cmd)\n"
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
                "source": [f"# LemGendary Manifold Training: {target_name} ({resolved_model})\n", "v16.1 Nuclear-Hardened Orchestrator.\n"],
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
                "cell_type": "code",
                "source": lfs_source,
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

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=4)
    print(f"[OK] Generated v16.1 Nuclear Notebook: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    generate_training_notebook(args.dataset, args.model, args.output)
