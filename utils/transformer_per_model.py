# IMPORTS *****************************************************************************************
# External libraries
import torch
from torchvision.transforms import v2


# UNI2-h ******************************************************************************************
def build_uni2h_transform():
    """
    Return the model-specific preprocessing transform for UNI2-h.
    """
    return v2.Compose([
        # STEP 1: Resize the tile to the model input size
        v2.Resize(
            size=(224, 224),
            interpolation=v2.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        # STEP 2: Normalize the tile with ImageNet channel statistics
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
