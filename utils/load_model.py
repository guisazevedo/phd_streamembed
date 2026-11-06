# IMPORTS *****************************************************************************************
# External libraries
from pathlib import Path
import torch
import timm
from timm.layers import SwiGLUPacked

# Internal imports ********************************************************************************
# Local folder holding the downloaded UNI2-h weights (pytorch_model.bin)
MODEL_DIR = Path("original_vit_model")

# Name used for the feature output subfolder and the encoder tag stored in each file
MODEL_NAME = "UNI2-h"

# Main Function ***********************************************************************************
def load_model(
    device: torch.device,
    model_dir: Path = MODEL_DIR,
) -> torch.nn.Module:
    """
    Load the UNI2-h model from local weights.

    * Arguments:
    device: The torch device used for inference
    model_dir: The local directory containing pytorch_model.bin
    """

    print(f"Loading model: {MODEL_NAME}")

    # STEP 1: Build the exact UNI2-h architecture
    model = timm.create_model(
        "vit_giant_patch14_224",
        pretrained=False,
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )

    # STEP 2: Load local checkpoint
    checkpoint_path = model_dir / "pytorch_model.bin"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"UNI2-h weights not found at {checkpoint_path}. Download pytorch_model.bin "
            "from huggingface.co/MahmoodLab/UNI2-h into that folder."
        )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    print(f"Model loaded: {MODEL_NAME}")

    # STEP 3: Move model to device and set inference mode
    model = model.to(device)
    print(f"Model moved to device: {device}")
    model.eval()

    return model
