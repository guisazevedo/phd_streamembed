import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from pathlib import Path
from huggingface_hub import snapshot_download

# Run from anywhere: the destination is resolved relative to the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

snapshot_download(
    repo_id="MahmoodLab/UNI2-h",
    local_dir=REPO_ROOT / "original_vit_model",
    token=True,
)
