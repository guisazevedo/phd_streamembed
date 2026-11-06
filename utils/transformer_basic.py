# IMPORTS *****************************************************************************************
# External libraries
import torch
from torchvision.transforms import v2


# Main function ***********************************************************************************
def build_basic_transform(
    apply_color_jitter: bool = False,
    apply_horizontal_flip: bool = False,
    apply_vertical_flip: bool = False,
    apply_rotation: bool = False,
    as_float: bool = False,
):
    """
    Return the shared preprocessing transform for the model.

    Only include model-agnostic operations here.

    * Arguments:
    apply_color_jitter: Whether to apply random color jitter augmentation
    apply_horizontal_flip: Whether to apply random horizontal flip augmentation
    apply_vertical_flip: Whether to apply random vertical flip augmentation
    apply_rotation: Whether to apply random rotation augmentation
    as_float: Whether to convert tiles to float32 in [0, 1] here. When False the tiles stay
        uint8, so batches are 4x smaller to copy to the GPU, where the float conversion happens
    """

    # STEP 1: Add required image conversion (and optional dtype conversion)
    transforms = [v2.ToImage()]

    if as_float:
        transforms.append(v2.ToDtype(torch.float32, scale=True))

    # TODO: Placeholders for future shared augmentations

    # STEP 2: Add optional color jitter augmentation
    if apply_color_jitter:
        transforms.append(
            v2.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            )
        )

    # STEP 3: Add optional horizontal flip augmentation
    if apply_horizontal_flip:
        transforms.append(
            v2.RandomHorizontalFlip(p=0.5)
        )

    # STEP 4: Add optional vertical flip augmentation
    if apply_vertical_flip:
        transforms.append(
            v2.RandomVerticalFlip(p=0.5)
        )

    # STEP 5: Add optional rotation augmentation
    if apply_rotation:
        transforms.append(
            v2.RandomRotation(
                degrees=(-180, 180),
            )
        )

    # STEP 6: Compose and return the transform pipeline
    return v2.Compose(transforms)
