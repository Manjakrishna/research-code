from pathlib import Path

p = Path("app/main.py")
s = p.read_text()

s = s.replace(
    "import torch\n",
    "import torch\nfrom huggingface_hub import hf_hub_download\n",
    1
)

old = '''    checkpoint_path = PROJECT_ROOT / config["training"]["checkpoint_dir"] / config["training"]["checkpoint_filename"]
    if not checkpoint_path.is_file():
        return None, None, config, device
'''

new = '''    checkpoint_path = PROJECT_ROOT / config["training"]["checkpoint_dir"] / config["training"]["checkpoint_filename"]

    if not checkpoint_path.is_file():
        try:
            checkpoint_path = Path(hf_hub_download(
                repo_id="krishnamanja/brain-tumor-model",
                filename="best_hybrid_model.pth"
            ))
        except Exception as e:
            st.error(f"Could not download model checkpoint: {e}")
            return None, None, config, device
'''

if old not in s:
    raise SystemExit("Expected code not found. No changes made.")

p.write_text(s.replace(old, new, 1))
print("SUCCESS: app/main.py updated")

