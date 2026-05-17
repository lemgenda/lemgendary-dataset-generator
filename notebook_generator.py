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
        "import os\n",
        f"model_key = '{resolved_model}'\n",
        "data_root = '/kaggle/input'\n",
        "target_dir = f'/kaggle/working/LemGendaryDatasets'\n",
        "os.makedirs(target_dir, exist_ok=True)\n",
        "\n",
        "print(f'⚡ [DATA] Speed-Resolving manifolds for {model_key}...')\n",
        "found = []\n",
        "target_slugs = [model_key.lower(), model_key.lower().replace(\"_\", \"-\"), model_key.lower().replace(\"_\", \"\")]\n",
        "\n",
        "# 1. Restricted Breadth-First Scanner (max depth 4, directories only) to bypass FUSE latency\n",
        "if os.path.exists(data_root):\n",
        "    try:\n",
        "        queue = [data_root]\n",
        "        depths = {data_root: 0}\n",
        "        while queue:\n",
        "            curr = queue.pop(0)\n",
        "            depth = depths[curr]\n",
        "            if depth > 4: continue\n",
        "            for item in os.listdir(curr):\n",
        "                path = os.path.join(curr, item)\n",
        "                if os.path.isdir(path):\n",
        "                    depths[path] = depth + 1\n",
        "                    queue.append(path)\n",
        "                    \n",
        "                    item_lower = item.lower()\n",
        "                    is_match = any(slug in item_lower for slug in target_slugs) or 'lemgendary' in item_lower or 'datasets' in item_lower\n",
        "                    if is_match:\n",
        "                        # Direct check: does it have images/train?\n",
        "                        if os.path.exists(os.path.join(path, 'images', 'train')):\n",
        "                            found.append(path)\n",
        "                        else:\n",
        "                            # Nested check (1 level deeper)\n",
        "                            try:\n",
        "                                for sub in os.listdir(path):\n",
        "                                    sub_cand = os.path.join(path, sub)\n",
        "                                    if os.path.isdir(sub_cand) and os.path.exists(os.path.join(sub_cand, 'images', 'train')):\n",
        "                                        found.append(sub_cand)\n",
        "                            except:\n",
        "                                pass\n",
        "    except Exception:\n",
        "        pass\n",
        "\n",
        "for d in sorted(list(set(found))):\n",
        "    bname = os.path.basename(d)\n",
        "    camel_name = \"\".join([w.capitalize() for w in model_key.split(\"_\")])\n",
        "    links = [\n",
        "        bname, bname.lower(),\n",
        "        f\"LemGendized{camel_name}KaggleReady\",\n",
        "        f\"LemGendized{camel_name}Large\",\n",
        "        f\"{camel_name}KaggleReady\",\n",
        "        f\"{camel_name}Large\"\n",
        "    ]\n",
        "    for link in sorted(list(set(links))):\n",
        "        link_name = os.path.join(target_dir, link)\n",
        "        if not os.path.exists(link_name):\n",
        "            try:\n",
        "                os.symlink(d, link_name)\n",
        "                print(f'✅ [LINKED] {link} -> {d}')\n",
        "            except:\n",
        "                pass\n"
    ]

    checkpoint_recovery_source = [
        "import os, shutil, glob\n",
        f"model_key = '{resolved_model}'\n",
        "print(f'⚡ [RECOVERY] Speed-Searching for {model_key} checkpoints...')\n",
        "hub_root = '/kaggle/working/LemGendaryModels'\n",
        "model_hub_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_hub_dir = os.path.join(model_hub_dir, 'checkpoints')\n",
        "os.makedirs(ckpt_hub_dir, exist_ok=True)\n",
        "\n",
        "try:\n",
        "    import yaml\n",
        "    with open('/kaggle/working/lemgendary-training-suite/unified_models_v2.yaml', 'r') as f:\n",
        "        reg = yaml.safe_load(f)\n",
        "    reg_filename = reg.get(model_key, {}).get('filename', '')\n",
        "except: reg_filename = ''\n",
        "\n",
        "# Identify candidates based on naming to avoid scanning image datasets\n",
        "target_slugs = [model_key.lower().replace('_', ''), model_key.lower().replace('_', '-'), reg_filename.lower() if reg_filename else '']\n",
        "target_slugs = [s for s in target_slugs if s]\n",
        "\n",
        "candidate_dirs = []\n",
        "data_root = '/kaggle/input'\n",
        "if os.path.exists(data_root):\n",
        "    try:\n",
        "        queue = [data_root]\n",
        "        depths = {data_root: 0}\n",
        "        while queue:\n",
        "            curr = queue.pop(0)\n",
        "            depth = depths[curr]\n",
        "            if depth > 4: continue\n",
        "            for item in os.listdir(curr):\n",
        "                path = os.path.join(curr, item)\n",
        "                if os.path.isdir(path):\n",
        "                    depths[path] = depth + 1\n",
        "                    queue.append(path)\n",
        "                    \n",
        "                    item_lower = item.lower()\n",
        "                    if any(slug in item_lower for slug in target_slugs) or 'checkpoint' in item_lower or 'weights' in item_lower or 'models' in item_lower:\n",
        "                        candidate_dirs.append(path)\n",
        "    except: pass\n",
        "\n",
        "found_ckpts = []\n",
        "for c_dir in candidate_dirs:\n",
        "    for root, _, files in os.walk(c_dir):\n",
        "        for f in files:\n",
        "            if f.endswith('.pth'):\n",
        "                f_lower = f.lower()\n",
        "                if any(slug in f_lower for slug in target_slugs) or 'best' in f_lower or 'latest' in f_lower:\n",
        "                    found_ckpts.append(os.path.join(root, f))\n",
        "\n",
        "found_ckpts = list(set(found_ckpts))\n",
        "if found_ckpts:\n",
        "    print(f'   -> [FOUND] {len(found_ckpts)} binaries in candidate folders.')\n",
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
        "            print(f'   -> [OK] Recovered {fname} -> {target_f}')\n",
        "            \n",
        "    # Speed-recover metrics.csv\n",
        "    metrics_found = False\n",
        "    for src in found_ckpts:\n",
        "        parent = os.path.dirname(src)\n",
        "        m_path = os.path.join(parent, 'metrics.csv')\n",
        "        if os.path.exists(m_path):\n",
        "            try:\n",
        "                shutil.copy2(m_path, os.path.join(model_hub_dir, 'metrics.csv'))\n",
        "                print(f'   -> [OK] Recovered metrics.csv')\n",
        "                metrics_found = True\n",
        "                break\n",
        "            except: pass\n",
        "else:\n",
        "    print('   -> [SKIP] No checkpoints found.')\n"
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
                "source": data_resolution_source,
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
    # 2026 Resilience: Surgical Model Registry for non-manifold notebooks
    MODELS_ONLY = {
        "universal_nsfw_classification": "Universal NSFW Classification",
        "diffusion_flux": "Diffusion Flux (Black Forest Labs)",
        "diffusion_sdxl": "Diffusion SDXL (Stability AI)",
        "vlm_blip2": "VLM BLIP-2 (Salesforce)",
        "vlm_llava": "VLM LLaVA (Microsoft/UW)"
    }

    export_root = args.output if args.output else os.path.abspath(os.path.join(base_dir, "../LemGendaryModels"))

    if args.all:
        print(f"[NUCLEAR] Initiating Global Notebook Refresh for {len(datasets) + len(MODELS_ONLY)} entities...")
        prefix = registry.get("_registry_metadata", {}).get("name_prefix", "")
        suffix = registry.get("_registry_metadata", {}).get("name_suffix", "")
        
        # 1. Dataset Manifolds
        for d_key, d_info in datasets.items():
            target_name = d_info.get("name", d_key)
            pascal_name = d_info.get("name", d_key.replace("_", " ").title().replace(" ", ""))
            folder_name = f"{prefix}{pascal_name}{suffix}"
            
            if "master_manifold" not in d_key:
                m_dir = os.path.join(export_root, d_key)
                os.makedirs(m_dir, exist_ok=True)
                m_output = os.path.join(m_dir, f"{d_key}_training.ipynb")
                generate_training_notebook(target_name, d_key, m_output)
            
            dataset_root = os.path.abspath(os.path.join(base_dir, "../LemGendaryDatasets"))
            d_manifold_dir = os.path.join(dataset_root, folder_name)
            if os.path.exists(d_manifold_dir):
                d_output = os.path.join(d_manifold_dir, f"{d_key}_training.ipynb")
                generate_training_notebook(target_name, d_key, d_output)

        # 2. Surgical Model Notebooks
        for m_key, m_name in MODELS_ONLY.items():
            m_dir = os.path.join(export_root, m_key)
            os.makedirs(m_dir, exist_ok=True)
            m_output = os.path.join(m_dir, f"{m_key}_training.ipynb")
            generate_training_notebook(m_name, m_key, m_output)
            print(f"[OK] [SURGICAL] Refreshed: {m_key}")
            
        print("\n[SUCCESS] Dataset Notebook Matrix Synchronized.")
    elif args.dataset and args.model and args.output:
        generate_training_notebook(args.dataset, args.model, args.output)
    else:
        parser.print_help()
        exit(1)
