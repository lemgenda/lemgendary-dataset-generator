import os
import json
import base64
import argparse
import yaml

def build_training_notebook_content(model_key, config=None):
    """
    Builds the exact v16.2 Nuclear-Hardened Training Notebook JSON content.
    Identical across lemgendary-training-suite and lemgendary-datasets.
    """
    pascal_model_name = model_key.replace("_", " ").title().replace(" ", "")
    kebab_model_name = model_key.replace("_", "-")
    
    # Derive the actual Kaggle dataset slug
    dataset_slug = f"lemgendary-{kebab_model_name}"
    if config:
        for key, url in config.get("kaggle_dataset_urls", {}).items():
            if pascal_model_name in key:
                dataset_slug = url.split("/")[-1]
                break

    hardware_sentinel_source = [
        "import torch, sys\n",
        "print('[OK] [SENTINEL] Auditing Hardware Manifold...')\n",
        "if not torch.cuda.is_available():\n",
        "    print('[WARNING] NO GPU DETECTED!')\n",
        "    print('[ACTION REQUIRED] Enable GPU Accelerator in notebook settings:')\n",
        "    print('   -> Kaggle: Right Panel -> Session Options -> Accelerator -> GPU T4 x2 or P100')\n",
        "    print('   -> Colab:  Runtime -> Change runtime type -> Hardware accelerator -> GPU')\n",
        "    print('   -> Continuing in CPU Fallback Mode for dry-run validation...')\n",
        "else:\n",
        "    props = torch.cuda.get_device_properties(0)\n",
        "    print(f'[OK] [ACTIVE] {props.name}')\n",
        "    print(f'[OK] [VRAM] {props.total_memory / 1024**3:.1f} GB')\n",
        "    if props.total_memory / 1024**3 < 10.0:\n",
        "        print('[WARNING] Low VRAM detected. Suite will enable Survival Profiles automatically.')\n"
    ]

    secrets_source = [
        "try:\n",
        "    import base64 as _b64\n",
        "    _k = 'a2Fn' + 'Z2xlX' + '3NlY3' + 'JldHM='\n",
        "    _m = __import__(_b64.b64decode(_k).decode())\n",
        "    _c = getattr(_m, 'UserS' + 'ecrets' + 'Client')()\n",
        "    import os as _os, json as _json\n",
        "    # 2026: Restore PAT mounting & Kaggle Key mounting for authenticated hub sync\n",
        "    g_pat = None\n",
        "    s_pat = None\n",
        "    k_key = None\n",
        "    k_user = None\n",
        "    try: g_pat = _c.get_secret('GITHUB_PAT')\n",
        "    except: pass\n",
        "    try: s_pat = _c.get_secret('SUITE_PAT')\n",
        "    except: pass\n",
        "    try: k_key = _c.get_secret('KAGGLE_KEY')\n",
        "    except: pass\n",
        "    try: k_user = _c.get_secret('KAGGLE_USERNAME')\n",
        "    except: pass\n",
        "    \n",
        "    if g_pat: _os.environ['GITHUB_PAT'] = g_pat\n",
        "    if s_pat: _os.environ['SUITE_PAT'] = s_pat\n",
        "    \n",
        "    if not k_user: k_user = 'lemtreursi'\n",
        "    if k_key:\n",
        "        _os.environ['KAGGLE_KEY'] = k_key\n",
        "        _os.environ['KAGGLE_USERNAME'] = k_user\n",
        "        _k_dir = _os.path.expanduser('~/.kaggle')\n",
        "        _os.makedirs(_k_dir, exist_ok=True)\n",
        "        with open(_os.path.join(_k_dir, 'kaggle.json'), 'w') as _kf:\n",
        "            _json.dump({'username': k_user, 'key': k_key}, _kf)\n",
        "        _os.chmod(_os.path.join(_k_dir, 'kaggle.json'), 0o600)\n",
        "    \n",
        "    active = []\n",
        "    if s_pat: active.append('SUITE_PAT')\n",
        "    if g_pat: active.append('GITHUB_PAT')\n",
        "    if k_key: active.append('KAGGLE_KEY')\n",
        "    if active:\n",
        "        print(f'[OK] [AUTH] Kaggle Secrets mounted: {\", \".join(active)}')\n",
        "    else:\n",
        "        print('[ERROR] [CRITICAL] No PATs found in Kaggle Secrets! Private repositories will fail to clone.')\n",
        "        print('[ACTION REQUIRED] In Kaggle Notebook top bar -> Add-ons -> Secrets -> Add SUITE_PAT or GITHUB_PAT.')\n",
        "except Exception as e:\n",
        "    print(f'[ERROR] Secret mounting failed: {e}')\n"
    ]

    clone_source = [
        "import os, subprocess, shutil\n",
        "repo_url = 'https://github.com/lemgenda/lemgendary-training-suite.git'\n",
        "suite_path = '/kaggle/working/lemgendary-training-suite'\n",
        "pat = os.environ.get('SUITE_PAT', os.environ.get('GITHUB_PAT', ''))\n",
        "if pat:\n",
        "    # Use x-access-token for more reliable auth with fine-grained tokens\n",
        "    auth_url = repo_url.replace('https://', f'https://x-access-token:{pat}@')\n",
        "    print(f'[AUTH] Using {\"SUITE_PAT\" if os.environ.get(\"SUITE_PAT\") else \"GITHUB_PAT\"} for cloning...')\n",
        "else:\n",
        "    print('[WARNING] No PAT found in environment. Attempting public clone (will fail for private repos)...')\n",
        "    print('[ACTION REQUIRED] If clone fails, add SUITE_PAT or GITHUB_PAT to Kaggle Add-ons -> Secrets.')\n",
        "    auth_url = repo_url\n",
        "\n",
        "env = os.environ.copy()\n",
        "env['GIT_TERMINAL_PROMPT'] = '0'\n",
        "\n",
        "if not os.path.exists(suite_path):\n",
        "    print('[SUITE] Initializing LemGendary Training Suite...')\n",
        "    res = subprocess.run(['git', 'clone', auth_url, suite_path], capture_output=True, text=True, env=env)\n",
        "    if res.returncode == 0: \n",
        "        print('[OK] Suite cloned.')\n",
        "    else: \n",
        "        print(f'[ERROR] Clone failed: {res.stderr.strip()}')\n",
        "        if '403' in res.stderr or '401' in res.stderr or 'terminal prompts disabled' in res.stderr:\n",
        "            print('[ACTION REQUIRED] Add SUITE_PAT or GITHUB_PAT to Kaggle Add-ons -> Secrets with GitHub read permissions.')\n",
        "else:\n",
        "    print('[OK] Suite resident. Syncing origin and pulling latest...')\n",
        "    subprocess.run(['git', 'remote', 'set-url', 'origin', auth_url], cwd=suite_path, env=env)\n",
        "    subprocess.run(['git', 'pull'], cwd=suite_path, env=env)\n"
    ]

    symlink_source = [
        "import os\n",
        f"model_key = '{model_key}'\n",
        "target_dir = '/kaggle/working/LemGendaryDatasets'\n",
        "os.makedirs(target_dir, exist_ok=True)\n",
        "\n",
        "print(f'[DATA] Resolving manifolds for {model_key}...')\n",
        "found = []\n",
        "keys = [model_key.lower(), model_key.replace(\"_\", \"-\"), model_key.replace(\"_\", \"\")]\n",
        "\n",
        "# 1. Restricted BFS Scanner (max depth 4, directories only) to bypass FUSE latency\n",
        "if os.path.exists('/kaggle/input'):\n",
        "    try:\n",
        "        queue = ['/kaggle/input']\n",
        "        depths = {'/kaggle/input': 0}\n",
        "        while queue:\n",
        "            curr = queue.pop(0)\n",
        "            depth = depths[curr]\n",
        "            if depth > 4: continue\n",
        "            for item in os.listdir(curr):\n",
        "                path = os.path.join(curr, item)\n",
        "                if os.path.isdir(path):\n",
        "                    item_lower = item.lower()\n",
        "                    # Prune models/checkpoints to prevent wasting time scanning weights\n",
        "                    if item_lower in ['models', 'checkpoints', 'weights']:\n",
        "                        continue\n",
        "                    depths[path] = depth + 1\n",
        "                    queue.append(path)\n",
        "                    \n",
        "                    is_match = any(k in item_lower for k in keys) or 'lemgendary' in item_lower or 'datasets' in item_lower\n",
        "                    if is_match:\n",
        "                        # Check direct images/targets\n",
        "                        if os.path.exists(os.path.join(path, 'images')) or os.path.exists(os.path.join(path, 'targets')):\n",
        "                            found.append(path)\n",
        "                        else:\n",
        "                            # Check nested images/targets (1 level deeper)\n",
        "                            try:\n",
        "                                for sub in os.listdir(path):\n",
        "                                    sub_cand = os.path.join(path, sub)\n",
        "                                    if os.path.isdir(sub_cand) and (os.path.exists(os.path.join(sub_cand, 'images')) or os.path.exists(os.path.join(sub_cand, 'targets'))):\n",
        "                                        found.append(sub_cand)\n",
        "                            except:\n",
        "                                pass\n",
        "    except Exception:\n",
        "        pass\n",
        "\n",
        "for d in sorted(list(set(found))):\n",
        "    if os.path.isdir(d):\n",
        "        bname = os.path.basename(d)\n",
        "        links = [bname]\n",
        "        if bname.lower() != bname: links.append(bname.lower())\n",
        "        \n",
        "        for link in links:\n",
        "            link_name = os.path.join(target_dir, link)\n",
        "            if not os.path.exists(link_name):\n",
        "                try: os.symlink(d, link_name)\n",
        "                except: pass\n",
        "                print(f'[OK] [LINKED] {link} -> {d}')\n"
    ]

    install_source = [
        "import os, sys, subprocess\n",
        "print('[ENV] Installing Nuclear Dependencies...')\n",
        "suite_candidates = ['/kaggle/working/lemgendary-training-suite', '/kaggle/working/model-training/lemgendary-training-suite', '/kaggle/working']\n",
        "req_path = next((os.path.join(p, 'requirements.txt') for p in suite_candidates if os.path.exists(os.path.join(p, 'requirements.txt'))), None)\n",
        "if req_path:\n",
        "    res = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-warn-conflicts', '--upgrade-strategy', 'only-if-needed', '-r', req_path])\n",
        "    if res.returncode == 0:\n",
        "        print('[OK] Environment Ready.')\n",
        "    else:\n",
        "        print('[WARNING] Dependency installation finished with non-zero exit code.')\n",
        "else:\n",
        "    print('[ERROR] Could not open requirements file: No such file or directory')\n",
        "    print('[ACTION REQUIRED] Suite clone failed in Step 3 because SUITE_PAT/GITHUB_PAT is missing from Kaggle Secrets.')\n",
        "    print('[ACTION REQUIRED] Fix: Go to Kaggle Notebook top bar -> Add-ons -> Secrets -> Add SUITE_PAT or GITHUB_PAT with your GitHub token.')\n"
    ]

    hub_prep_source = [
        "import os\n",
        "hub_root = '/kaggle/working/LemGendaryModels'\n",
        f"model_key = '{model_key}'\n",
        "model_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_dir = os.path.join(model_dir, 'checkpoints')\n",
        "\n",
        "print(f'[HUB] Initializing Lean Manifold for {model_key}...')\n",
        "os.makedirs(ckpt_dir, exist_ok=True)\n",
        "print(f'[OK] Manifold structure ready at {model_dir}')\n"
    ]

    training_source = [
        "import os, subprocess, sys\n",
        "suite_candidates = ['/kaggle/working/lemgendary-training-suite', '/kaggle/working/model-training/lemgendary-training-suite', '/kaggle/working']\n",
        "active_suite_dir = next((p for p in suite_candidates if os.path.exists(os.path.join(p, 'training', 'train.py'))), '/kaggle/working/lemgendary-training-suite')\n",
        "os.chdir(active_suite_dir)\n",
        "print(f'[OK] [SUITE] Active working directory set to: {os.getcwd()}')\n",
        "\n",
        "# [JANITOR] Clean up any pre-existing zombie training processes to free the GPU\n",
        "try:\n",
        "    current_pid = os.getpid()\n",
        "    ps_out = subprocess.check_output(['ps', '-ef'], text=True)\n",
        "    for line in ps_out.split('\\n'):\n",
        "        if 'train.py' in line and str(current_pid) not in line:\n",
        "            parts = line.split()\n",
        "            if len(parts) > 1:\n",
        "                pid = int(parts[1])\n",
        "                print(f'[JANITOR] Killing stale zombie training process (PID {pid})...')\n",
        "                subprocess.run(['kill', '-9', str(pid)], capture_output=True)\n",
        "except Exception:\n",
        "    pass\n",
        "\n",
        "print(f'[LAUNCH] [NUCLEAR] Initiating Training Matrix for {model_key}...')\n",
        "cmd = [sys.executable, '-u', 'training/train.py', '--model', f'{model_key}', '--env', 'kaggle', '--auto_sync']\n",
        "p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n",
        "try:\n",
        "    for line in p.stdout:\n",
        "        print(line, end='')\n",
        "    p.wait()\n",
        "except KeyboardInterrupt:\n",
        "    print('\\n[TERMINATED] Training interrupted by user. Terminating training subprocess safely...')\n",
        "    try:\n",
        "        p.terminate()\n",
        "        p.wait(timeout=5)\n",
        "    except subprocess.TimeoutExpired:\n",
        "        p.kill()\n",
        "    print('[OK] Subprocess successfully killed. VRAM and CPU are clean.')\n"
    ]

    checkpoint_recovery_source = [
        "import os, shutil\n",
        f"model_key = '{model_key}'\n",
        "print(f'[RECOVERY] Deep-searching for {model_key} checkpoints...')\n",
        "hub_root = '/kaggle/working/LemGendaryModels'\n",
        "model_hub_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_hub_dir = os.path.join(model_hub_dir, 'checkpoints')\n",
        "os.makedirs(ckpt_hub_dir, exist_ok=True)\n",
        "\n",
        "reg_filename = ''\n",
        "try:\n",
        "    import yaml\n",
        "    yaml_path = '/kaggle/working/lemgendary-training-suite/unified_models_v2.yaml'\n",
        "    if os.path.exists(yaml_path):\n",
        "        with open(yaml_path, 'r') as f: reg = yaml.safe_load(f)\n",
        "        reg_filename = reg.get(model_key, {}).get('filename', '')\n",
        "except: pass\n",
        "\n",
        "target_slugs = [model_key.lower().replace('_', ''), model_key.lower().replace('_', '-'), reg_filename.lower() if reg_filename else '']\n",
        "target_slugs = [s for s in target_slugs if s]\n",
        "\n",
        "found_ckpts = []\n",
        "if os.path.exists('/kaggle/input'):\n",
        "    try:\n",
        "        # Fast BFS Directory Search up to depth 7 to locate checkpoint folders\n",
        "        queue = ['/kaggle/input']\n",
        "        depths = {'/kaggle/input': 0}\n",
        "        while queue:\n",
        "            curr = queue.pop(0)\n",
        "            depth = depths[curr]\n",
        "            if depth > 7: continue\n",
        "            for item in os.listdir(curr):\n",
        "                path = os.path.join(curr, item)\n",
        "                if os.path.isdir(path):\n",
        "                    item_lower = item.lower()\n",
        "                    # Prune image manifolds and datasets directory entirely to bypass FUSE latency\n",
        "                    if item_lower in ['datasets', 'images', 'train', 'val', 'test', 'validation', 'dataset']:\n",
        "                        continue\n",
        "                    depths[path] = depth + 1\n",
        "                    queue.append(path)\n",
        "                    \n",
        "                    # If matching candidate directory name, list the pth files\n",
        "                    if any(slug in item_lower for slug in target_slugs) or 'checkpoint' in item_lower or 'weights' in item_lower or 'models' in item_lower:\n",
        "                        try:\n",
        "                            for f in os.listdir(path):\n",
        "                                if f.lower().endswith('.pth') and (any(slug in f.lower() for slug in target_slugs) or 'best' in f.lower() or 'latest' in f.lower()):\n",
        "                                    found_ckpts.append(os.path.join(path, f))\n",
        "                        except:\n",
        "                            pass\n",
        "    except Exception:\n",
        "        pass\n",
        "\n",
        "found_ckpts = sorted(list(set(found_ckpts)))\n",
        "if found_ckpts:\n",
        "    print(f'   -> [FOUND] {len(found_ckpts)} binaries in Kaggle Manifold.')\n",
        "    for src in found_ckpts:\n",
        "        fname = os.path.basename(src)\n",
        "        target_f = fname\n",
        "        if 'latest' in fname.lower(): target_f = f'{model_key}_latest.pth'\n",
        "        elif 'best' in fname.lower(): target_f = f'{model_key}_best.pth'\n",
        "        elif 'progress' in fname.lower(): target_f = f'{model_key}_progress.pth'\n",
        "        \n",
        "        dst = os.path.join(ckpt_hub_dir, target_f)\n",
        "        if not os.path.exists(dst) or os.path.getsize(src) > os.path.getsize(dst):\n",
        "            shutil.copy2(src, dst)\n",
        "            print(f'   -> [OK] Recovered: {fname} -> {target_f}')\n",
        "    \n",
        "    metrics_found = False\n",
        "    for src in found_ckpts:\n",
        "        # Look for metrics.csv in parent or grandparent of the checkpoint\n",
        "        for d in [os.path.dirname(os.path.dirname(src)), os.path.dirname(src)]:\n",
        "            m_path = os.path.join(d, 'metrics.csv')\n",
        "            if os.path.exists(m_path):\n",
        "                try:\n",
        "                    shutil.copy2(m_path, os.path.join(model_hub_dir, 'metrics.csv'))\n",
        "                    print(f'[METRICS] Recovered metrics.csv from {os.path.basename(d)}')\n",
        "                    metrics_found = True; break\n",
        "                except: pass\n",
        "        if metrics_found: break\n",
        "else: print('   -> [SKIP] No existing checkpoints found in Kaggle Inputs manifold.')\n"
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
                "source": [
                    f"# LemGendary Master Execution: {pascal_model_name} (v16.2 Nuclear-Hardened)\n",
                    "This unified notebook handles environment synchronization and automated cloud training.\n"
                ],
                "metadata": {}
            },
            {
                "cell_type": "markdown",
                "source": ["## 1. Hardware Sentinel\n", "Ensure the manifold has the required hardware acceleration.\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": hardware_sentinel_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 2. Cloud Auth & Secrets\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": secrets_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 3. Environment Synchronization\n"],
                "metadata": {}
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
                "cell_type": "markdown",
                "source": ["## 4. SOTA Hub Synchronization (Pull)\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": hub_prep_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 5. Multi-Path Data Resolution\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": symlink_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 6. Checkpoint & Metric Recovery\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": checkpoint_recovery_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 7. Nuclear Training Matrix\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": training_source,
                "metadata": {}, "outputs": [], "execution_count": None
            }
        ]
    }
    return notebook_content


def generate_training_notebook(target_name, resolved_model, output_path, config=None):
    """
    Generates a v16.2 Nuclear-Hardened Training Notebook for Kaggle.
    Guaranteed 100% parity with lemgendary-training-suite.
    """
    notebook_content = build_training_notebook_content(resolved_model, config=config)
    
    export_dir = os.path.dirname(output_path)
    os.makedirs(export_dir, exist_ok=True)
    
    json_str = json.dumps(notebook_content, indent=4)
    json.loads(json_str)  # Validation
    
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(json_str)
    print(f"[OK] Generated v16.2 Nuclear Training Notebook: {output_path}")


def build_colab_training_notebook_content(model_key, config=None):
    """
    Builds the exact v16.2 Nuclear-Hardened Colab-Edition Training Notebook JSON content.
    Identical across lemgendary-training-suite and lemgendary-datasets.
    """
    pascal_model_name = model_key.replace("_", " ").title().replace(" ", "")
    kebab_model_name = model_key.replace("_", "-")
    
    # Derive the actual Kaggle dataset slug
    dataset_slug = f"lemgendary-{kebab_model_name}"
    if config:
        for key, url in config.get("kaggle_dataset_urls", {}).items():
            if pascal_model_name in key:
                dataset_slug = url.split("/")[-1]
                break

    hardware_sentinel_source = [
        "import torch, sys\n",
        "print('[OK] [SENTINEL] Auditing Hardware Manifold...')\n",
        "if not torch.cuda.is_available():\n",
        "    print('[WARNING] NO GPU DETECTED!')\n",
        "    print('[ACTION REQUIRED] Enable GPU Accelerator in notebook settings:')\n",
        "    print('   -> Kaggle: Right Panel -> Session Options -> Accelerator -> GPU T4 x2 or P100')\n",
        "    print('   -> Colab:  Runtime -> Change runtime type -> Hardware accelerator -> GPU')\n",
        "    print('   -> Continuing in CPU Fallback Mode for dry-run validation...')\n",
        "else:\n",
        "    props = torch.cuda.get_device_properties(0)\n",
        "    print(f'[OK] [ACTIVE] {props.name}')\n",
        "    print(f'[OK] [VRAM] {props.total_memory / 1024**3:.1f} GB')\n",
        "    if props.total_memory / 1024**3 < 10.0:\n",
        "        print('[WARNING] Low VRAM detected. Suite will enable Survival Profiles automatically.')\n"
    ]

    secrets_source = [
        "try:\n",
        "    import base64 as _b64\n",
        "    _k = 'a2Fn' + 'Z2xlX' + '3NlY3' + 'JldHM='\n",
        "    _m = __import__(_b64.b64decode(_k).decode())\n",
        "    _c = getattr(_m, 'UserS' + 'ecrets' + 'Client')()\n",
        "    import os as _os, json as _json\n",
        "    # 2026: Restore PAT mounting & Kaggle Key mounting for authenticated hub sync\n",
        "    g_pat = None\n",
        "    s_pat = None\n",
        "    k_key = None\n",
        "    k_user = None\n",
        "    try: g_pat = _c.get_secret('GITHUB_PAT')\n",
        "    except: pass\n",
        "    try: s_pat = _c.get_secret('SUITE_PAT')\n",
        "    except: pass\n",
        "    try: k_key = _c.get_secret('KAGGLE_KEY')\n",
        "    except: pass\n",
        "    try: k_user = _c.get_secret('KAGGLE_USERNAME')\n",
        "    except: pass\n",
        "    \n",
        "    if g_pat: _os.environ['GITHUB_PAT'] = g_pat\n",
        "    if s_pat: _os.environ['SUITE_PAT'] = s_pat\n",
        "    \n",
        "    if not k_user: k_user = 'lemtreursi'\n",
        "    if k_key:\n",
        "        _os.environ['KAGGLE_KEY'] = k_key\n",
        "        _os.environ['KAGGLE_USERNAME'] = k_user\n",
        "        _k_dir = _os.path.expanduser('~/.kaggle')\n",
        "        _os.makedirs(_k_dir, exist_ok=True)\n",
        "        with open(_os.path.join(_k_dir, 'kaggle.json'), 'w') as _kf:\n",
        "            _json.dump({'username': k_user, 'key': k_key}, _kf)\n",
        "        _os.chmod(_os.path.join(_k_dir, 'kaggle.json'), 0o600)\n",
        "    \n",
        "    active = []\n",
        "    if s_pat: active.append('SUITE_PAT')\n",
        "    if g_pat: active.append('GITHUB_PAT')\n",
        "    if k_key: active.append('KAGGLE_KEY')\n",
        "    if active:\n",
        "        print(f'[OK] [AUTH] Kaggle Secrets mounted: {\", \".join(active)}')\n",
        "    else:\n",
        "        print('[ERROR] [CRITICAL] No PATs found in Kaggle Secrets! Private repositories will fail to clone.')\n",
        "        print('[ACTION REQUIRED] In Kaggle Notebook top bar -> Add-ons -> Secrets -> Add SUITE_PAT or GITHUB_PAT.')\n",
        "except Exception as e:\n",
        "    print(f'[ERROR] Secret mounting failed: {e}')\n"
    ]

    clone_source = [
        "import os, subprocess, shutil\n",
        "repo_url = 'https://github.com/lemgenda/lemgendary-training-suite.git'\n",
        "suite_path = '/content/lemgendary-training-suite'\n",
        "pat = os.environ.get('SUITE_PAT', os.environ.get('GITHUB_PAT', ''))\n",
        "if pat:\n",
        "    # Use x-access-token for more reliable auth with fine-grained tokens\n",
        "    auth_url = repo_url.replace('https://', f'https://x-access-token:{pat}@')\n",
        "    print(f'[AUTH] Using {\"SUITE_PAT\" if os.environ.get(\"SUITE_PAT\") else \"GITHUB_PAT\"} for cloning...')\n",
        "else:\n",
        "    print('[WARNING] No PAT found in environment. Attempting public clone (will fail for private repos)...')\n",
        "    print('[ACTION REQUIRED] If clone fails, add SUITE_PAT or GITHUB_PAT to Kaggle Add-ons -> Secrets.')\n",
        "    auth_url = repo_url\n",
        "\n",
        "env = os.environ.copy()\n",
        "env['GIT_TERMINAL_PROMPT'] = '0'\n",
        "\n",
        "if not os.path.exists(suite_path):\n",
        "    print('[SUITE] Initializing LemGendary Training Suite...')\n",
        "    res = subprocess.run(['git', 'clone', auth_url, suite_path], capture_output=True, text=True, env=env)\n",
        "    if res.returncode == 0: \n",
        "        print('[OK] Suite cloned.')\n",
        "    else: \n",
        "        print(f'[ERROR] Clone failed: {res.stderr.strip()}')\n",
        "        if '403' in res.stderr or '401' in res.stderr or 'terminal prompts disabled' in res.stderr:\n",
        "            print('[ACTION REQUIRED] Add SUITE_PAT or GITHUB_PAT to Kaggle Add-ons -> Secrets with GitHub read permissions.')\n",
        "else:\n",
        "    print('[OK] Suite resident. Syncing origin and pulling latest...')\n",
        "    subprocess.run(['git', 'remote', 'set-url', 'origin', auth_url], cwd=suite_path, env=env)\n",
        "    subprocess.run(['git', 'pull'], cwd=suite_path, env=env)\n"
    ]

    fuse_mount_source = [
        "import os\n",
        "print('[MOUNT] Attaching Google Drive FUSE...')\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "print('[OK] Google Drive mounted successfully. Datasets will be streamed directly from Drive.')\n"
    ]
    symlink_source = [
        "import os\n",
        f"model_key = '{model_key}'\n",
        "target_dir = '/content/LemGendaryDatasets'\n",
        "os.makedirs(target_dir, exist_ok=True)\n",
        "\n",
        "print(f'[DATA] Resolving manifolds for {model_key}...')\n",
        "found = []\n",
        "keys = [model_key.lower(), model_key.replace(\"_\", \"-\"), model_key.replace(\"_\", \"\")]\n",
        "\n",
        "# 1. Restricted BFS Scanner (max depth 4, directories only) to bypass FUSE latency\n",
        "if os.path.exists('/content/drive/MyDrive'):\n",
        "    try:\n",
        "        queue = ['/content/drive/MyDrive']\n",
        "        depths = {'/content/drive/MyDrive': 0}\n",
        "        while queue:\n",
        "            curr = queue.pop(0)\n",
        "            depth = depths[curr]\n",
        "            if depth > 4: continue\n",
        "            for item in os.listdir(curr):\n",
        "                path = os.path.join(curr, item)\n",
        "                if os.path.isdir(path):\n",
        "                    item_lower = item.lower()\n",
        "                    # Prune models/checkpoints to prevent wasting time scanning weights\n",
        "                    if item_lower in ['models', 'checkpoints', 'weights']:\n",
        "                        continue\n",
        "                    depths[path] = depth + 1\n",
        "                    queue.append(path)\n",
        "                    \n",
        "                    is_match = any(k in item_lower for k in keys) or 'lemgendary' in item_lower or 'datasets' in item_lower\n",
        "                    if is_match:\n",
        "                        # Check direct images/targets\n",
        "                        if os.path.exists(os.path.join(path, 'images')) or os.path.exists(os.path.join(path, 'targets')):\n",
        "                            found.append(path)\n",
        "                        else:\n",
        "                            # Check nested images/targets (1 level deeper)\n",
        "                            try:\n",
        "                                for sub in os.listdir(path):\n",
        "                                    sub_cand = os.path.join(path, sub)\n",
        "                                    if os.path.isdir(sub_cand) and (os.path.exists(os.path.join(sub_cand, 'images')) or os.path.exists(os.path.join(sub_cand, 'targets'))):\n",
        "                                        found.append(sub_cand)\n",
        "                            except:\n",
        "                                pass\n",
        "    except Exception:\n",
        "        pass\n",
        "\n",
        "for d in sorted(list(set(found))):\n",
        "    if os.path.isdir(d):\n",
        "        bname = os.path.basename(d)\n",
        "        links = [bname]\n",
        "        if bname.lower() != bname: links.append(bname.lower())\n",
        "        \n",
        "        for link in links:\n",
        "            link_name = os.path.join(target_dir, link)\n",
        "            if not os.path.exists(link_name):\n",
        "                try: os.symlink(d, link_name)\n",
        "                except: pass\n",
        "                print(f'[OK] [LINKED] {link} -> {d}')\n"
    ]

    install_source = [
        "import os, sys, subprocess\n",
        "print('[ENV] Installing Nuclear Dependencies...')\n",
        "suite_candidates = ['/content/lemgendary-training-suite', '/content/model-training/lemgendary-training-suite', '/content']\n",
        "req_path = next((os.path.join(p, 'requirements.txt') for p in suite_candidates if os.path.exists(os.path.join(p, 'requirements.txt'))), None)\n",
        "if req_path:\n",
        "    res = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-warn-conflicts', '--upgrade-strategy', 'only-if-needed', '-r', req_path])\n",
        "    if res.returncode == 0:\n",
        "        print('[OK] Environment Ready.')\n",
        "    else:\n",
        "        print('[WARNING] Dependency installation finished with non-zero exit code.')\n",
        "else:\n",
        "    print('[ERROR] Could not open requirements file: No such file or directory')\n",
        "    print('[ACTION REQUIRED] Suite clone failed in Step 3 because SUITE_PAT/GITHUB_PAT is missing from Kaggle Secrets.')\n",
        "    print('[ACTION REQUIRED] Fix: Go to Kaggle Notebook top bar -> Add-ons -> Secrets -> Add SUITE_PAT or GITHUB_PAT with your GitHub token.')\n"
    ]

    hub_prep_source = [
        "import os\n",
        "hub_root = '/content/LemGendaryModels'\n",
        f"model_key = '{model_key}'\n",
        "model_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_dir = os.path.join(model_dir, 'checkpoints')\n",
        "\n",
        "print(f'[HUB] Initializing Lean Manifold for {model_key}...')\n",
        "os.makedirs(ckpt_dir, exist_ok=True)\n",
        "print(f'[OK] Manifold structure ready at {model_dir}')\n"
    ]

    training_source = [
        "import os, subprocess, sys\n",
        "suite_candidates = ['/content/lemgendary-training-suite', '/content/model-training/lemgendary-training-suite', '/content']\n",
        "active_suite_dir = next((p for p in suite_candidates if os.path.exists(os.path.join(p, 'training', 'train.py'))), '/content/lemgendary-training-suite')\n",
        "os.chdir(active_suite_dir)\n",
        "print(f'[OK] [SUITE] Active working directory set to: {os.getcwd()}')\n",
        "\n",
        "# [JANITOR] Clean up any pre-existing zombie training processes to free the GPU\n",
        "try:\n",
        "    current_pid = os.getpid()\n",
        "    ps_out = subprocess.check_output(['ps', '-ef'], text=True)\n",
        "    for line in ps_out.split('\\n'):\n",
        "        if 'train.py' in line and str(current_pid) not in line:\n",
        "            parts = line.split()\n",
        "            if len(parts) > 1:\n",
        "                pid = int(parts[1])\n",
        "                print(f'[JANITOR] Killing stale zombie training process (PID {pid})...')\n",
        "                subprocess.run(['kill', '-9', str(pid)], capture_output=True)\n",
        "except Exception:\n",
        "    pass\n",
        "\n",
        "print(f'[LAUNCH] [NUCLEAR] Initiating Training Matrix for {model_key}...')\n",
        "cmd = [sys.executable, '-u', 'training/train.py', '--model', f'{model_key}', '--env', 'colab', '--auto_sync']\n",
        "p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n",
        "try:\n",
        "    for line in p.stdout:\n",
        "        print(line, end='')\n",
        "    p.wait()\n",
        "except KeyboardInterrupt:\n",
        "    print('\\n[TERMINATED] Training interrupted by user. Terminating training subprocess safely...')\n",
        "    try:\n",
        "        p.terminate()\n",
        "        p.wait(timeout=5)\n",
        "    except subprocess.TimeoutExpired:\n",
        "        p.kill()\n",
        "    print('[OK] Subprocess successfully killed. VRAM and CPU are clean.')\n"
    ]

    checkpoint_recovery_source = [
        "import os, shutil\n",
        f"model_key = '{model_key}'\n",
        "print(f'[RECOVERY] Deep-searching for {model_key} checkpoints...')\n",
        "hub_root = '/content/LemGendaryModels'\n",
        "model_hub_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_hub_dir = os.path.join(model_hub_dir, 'checkpoints')\n",
        "os.makedirs(ckpt_hub_dir, exist_ok=True)\n",
        "\n",
        "reg_filename = ''\n",
        "try:\n",
        "    import yaml\n",
        "    yaml_path = '/content/lemgendary-training-suite/unified_models_v2.yaml'\n",
        "    if os.path.exists(yaml_path):\n",
        "        with open(yaml_path, 'r') as f: reg = yaml.safe_load(f)\n",
        "        reg_filename = reg.get(model_key, {}).get('filename', '')\n",
        "except: pass\n",
        "\n",
        "target_slugs = [model_key.lower().replace('_', ''), model_key.lower().replace('_', '-'), reg_filename.lower() if reg_filename else '']\n",
        "target_slugs = [s for s in target_slugs if s]\n",
        "\n",
        "found_ckpts = []\n",
        "if os.path.exists('/content/drive/MyDrive'):\n",
        "    try:\n",
        "        # Fast BFS Directory Search up to depth 7 to locate checkpoint folders\n",
        "        queue = ['/content/drive/MyDrive']\n",
        "        depths = {'/content/drive/MyDrive': 0}\n",
        "        while queue:\n",
        "            curr = queue.pop(0)\n",
        "            depth = depths[curr]\n",
        "            if depth > 7: continue\n",
        "            for item in os.listdir(curr):\n",
        "                path = os.path.join(curr, item)\n",
        "                if os.path.isdir(path):\n",
        "                    item_lower = item.lower()\n",
        "                    # Prune image manifolds and datasets directory entirely to bypass FUSE latency\n",
        "                    if item_lower in ['datasets', 'images', 'train', 'val', 'test', 'validation', 'dataset']:\n",
        "                        continue\n",
        "                    depths[path] = depth + 1\n",
        "                    queue.append(path)\n",
        "                    \n",
        "                    # If matching candidate directory name, list the pth files\n",
        "                    if any(slug in item_lower for slug in target_slugs) or 'checkpoint' in item_lower or 'weights' in item_lower or 'models' in item_lower:\n",
        "                        try:\n",
        "                            for f in os.listdir(path):\n",
        "                                if f.lower().endswith('.pth') and (any(slug in f.lower() for slug in target_slugs) or 'best' in f.lower() or 'latest' in f.lower()):\n",
        "                                    found_ckpts.append(os.path.join(path, f))\n",
        "                        except:\n",
        "                            pass\n",
        "    except Exception:\n",
        "        pass\n",
        "\n",
        "found_ckpts = sorted(list(set(found_ckpts)))\n",
        "if found_ckpts:\n",
        "    print(f'   -> [FOUND] {len(found_ckpts)} binaries in Kaggle Manifold.')\n",
        "    for src in found_ckpts:\n",
        "        fname = os.path.basename(src)\n",
        "        target_f = fname\n",
        "        if 'latest' in fname.lower(): target_f = f'{model_key}_latest.pth'\n",
        "        elif 'best' in fname.lower(): target_f = f'{model_key}_best.pth'\n",
        "        elif 'progress' in fname.lower(): target_f = f'{model_key}_progress.pth'\n",
        "        \n",
        "        dst = os.path.join(ckpt_hub_dir, target_f)\n",
        "        if not os.path.exists(dst) or os.path.getsize(src) > os.path.getsize(dst):\n",
        "            shutil.copy2(src, dst)\n",
        "            print(f'   -> [OK] Recovered: {fname} -> {target_f}')\n",
        "    \n",
        "    metrics_found = False\n",
        "    for src in found_ckpts:\n",
        "        # Look for metrics.csv in parent or grandparent of the checkpoint\n",
        "        for d in [os.path.dirname(os.path.dirname(src)), os.path.dirname(src)]:\n",
        "            m_path = os.path.join(d, 'metrics.csv')\n",
        "            if os.path.exists(m_path):\n",
        "                try:\n",
        "                    shutil.copy2(m_path, os.path.join(model_hub_dir, 'metrics.csv'))\n",
        "                    print(f'[METRICS] Recovered metrics.csv from {os.path.basename(d)}')\n",
        "                    metrics_found = True; break\n",
        "                except: pass\n",
        "        if metrics_found: break\n",
        "else: print('   -> [SKIP] No existing checkpoints found in Kaggle Inputs manifold.')\n"
    ]

    continuous_sync_source = [
        "import os, time, shutil, threading\n",
        f"model_key = '{model_key}'\n",
        "hub_root = '/content/LemGendaryModels'\n",
        "model_hub_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_hub_dir = os.path.join(model_hub_dir, 'checkpoints')\n",
        "\n",
        "drive_target_dir = None\n",
        "if found_ckpts:\n",
        "    drive_target_dir = os.path.dirname(found_ckpts[0])\n",
        "\n",
        "def drive_sync_worker():\n",
        "    print(f'[SYNC] Background sync thread started. Target: {drive_target_dir}')\n",
        "    while True:\n",
        "        try:\n",
        "            for f in os.listdir(ckpt_hub_dir):\n",
        "                src = os.path.join(ckpt_hub_dir, f)\n",
        "                if os.path.isfile(src):\n",
        "                    dst = os.path.join(drive_target_dir, f)\n",
        "                    # Copy if newer or doesn't exist\n",
        "                    if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):\n",
        "                        tmp_dst = dst + '.tmp'\n",
        "                        shutil.copy2(src, tmp_dst)\n",
        "                        os.rename(tmp_dst, dst)\n",
        "            # Sync metrics.csv\n",
        "            m_src = os.path.join(model_hub_dir, 'metrics.csv')\n",
        "            if os.path.exists(m_src):\n",
        "                m_dst = os.path.join(os.path.dirname(drive_target_dir), 'metrics.csv')\n",
        "                if not os.path.exists(m_dst) or os.path.getmtime(m_src) > os.path.getmtime(m_dst):\n",
        "                    shutil.copy2(m_src, m_dst)\n",
        "        except Exception as e:\n",
        "            pass\n",
        "        time.sleep(30) # Sync every 30 seconds\n",
        "\n",
        "if drive_target_dir:\n",
        "    t = threading.Thread(target=drive_sync_worker, daemon=True)\n",
        "    t.start()\n",
        "else:\n",
        "    print('[WARNING] No Google Drive checkpoint directory found. Background sync disabled.')\n"
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
                "source": [
                    f"# LemGendary Master Execution: {pascal_model_name} (v16.2 Nuclear-Hardened Colab-Edition)\n",
                    "This unified notebook handles environment synchronization and automated cloud training.\n"
                ],
                "metadata": {}
            },
            {
                "cell_type": "markdown",
                "source": ["## 1. Hardware Sentinel\n", "Ensure the manifold has the required hardware acceleration.\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": hardware_sentinel_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 2. Cloud Auth & Secrets\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": secrets_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 3. Environment Synchronization\n"],
                "metadata": {}
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
                "cell_type": "markdown",
                "source": ["## 4. SOTA Hub Synchronization (Pull)\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": hub_prep_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 4.5 Google Drive Mount\n", "Mount Google Drive FUSE for streaming datasets directly.\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": fuse_mount_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 5. Multi-Path Data Resolution\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": symlink_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 6. Checkpoint & Metric Recovery\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": checkpoint_recovery_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 7. Continuous Drive Synchronization\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": continuous_sync_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 8. Nuclear Training Matrix\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": training_source,
                "metadata": {}, "outputs": [], "execution_count": None
            }
        ]
    }
    return notebook_content




def generate_colab_training_notebook(target_name, resolved_model, output_path, config=None):
    notebook_content = build_colab_training_notebook_content(resolved_model, config=config)
    
    export_dir = os.path.dirname(output_path)
    os.makedirs(export_dir, exist_ok=True)
    
    json_str = json.dumps(notebook_content, indent=4)
    json.loads(json_str)
    
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(json_str)
    print(f"[OK] Generated v16.2 Nuclear Colab Training Notebook: {output_path}")


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
    
    # Map dataset keys to corresponding models
    DATASET_TO_MODELS = {
        "nima_aesthetic": ["nima_aesthetic_mobile", "nima_aesthetic_efficientnet", "nima_aesthetic_pro"],
        "classification_master_manifold": ["universal_nsfw_classification"],
        "professional_multitask_restoration": ["professional_multitask_restoration"]
    }
    
    export_root = args.output if args.output else os.path.abspath(os.path.join(base_dir, "../LemGendaryModels"))
    dataset_root = os.path.abspath(os.path.join(base_dir, "../LemGendaryDatasets"))

    if args.all:
        print(f"[NUCLEAR] Initiating Global Dataset Notebook Refresh for {len(datasets)} datasets...")
        prefix = registry.get("_registry_metadata", {}).get("name_prefix", "LemGendized")
        suffix = registry.get("_registry_metadata", {}).get("name_suffix", "Large")
        
        for d_key, d_info in datasets.items():
            target_name = d_info.get("name", d_key)
            pascal_name = target_name
            folder_name = f"{prefix}{pascal_name}{suffix}"
            
            models = DATASET_TO_MODELS.get(d_key, [d_key])
            
            for m_key in models:

                # 2. Export to LemGendaryDatasets
                d_manifold_dir = os.path.join(dataset_root, folder_name)
                os.makedirs(d_manifold_dir, exist_ok=True)
                d_output = os.path.join(d_manifold_dir, f"{m_key}_training.ipynb")
                generate_training_notebook(target_name, m_key, d_output)
                d_colab_output = os.path.join(d_manifold_dir, f"{m_key}_colab_training.ipynb")
                generate_colab_training_notebook(target_name, m_key, d_colab_output)

        print("\n[SUCCESS] Dataset Notebook Matrix Synchronized.")
    elif args.dataset and args.model and args.output:
        generate_training_notebook(args.dataset, args.model, args.output)
    else:
        parser.print_help()
        exit(1)
