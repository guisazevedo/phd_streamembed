# IMPORTS *****************************************************************************************
# External libraries
from concurrent.futures import ThreadPoolExecutor
import torch
from tqdm.auto import tqdm
from pathlib import Path


def _prepare_model(model, device):
    """Move the model to device and set eval mode."""
    model = model.to(device)
    model.eval()
    return model


def _move_images_to_device(batch, device):
    """
    Extract images from the batch and move them to the inference device.

    uint8 tiles are copied as-is and converted to float32 in [0, 1] on the device.
    """
    images = batch["image"].to(device, non_blocking=True)

    if images.dtype == torch.uint8:
        images = images.float().div_(255)

    return images


def _run_model(images, transform, model, device):
    """Apply the model transform and run the model."""
    images = transform(images)

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            if hasattr(model, "forward_features"):
                return model.forward_features(images)

            return model(images)


def _assert_square_patch_grid(patch_features):
    """Raise if the patch token count is not a perfect square (e.g. 256 = 16x16)."""

    num_patches = patch_features.shape[1]
    grid_size = round(num_patches ** 0.5)

    if grid_size * grid_size != num_patches:
        raise ValueError(
            f"Expected a square patch grid, got {num_patches} patch tokens "
            f"(shape {tuple(patch_features.shape)}). This usually means prefix "
            "tokens (CLS/register) were not fully stripped."
        )


def _normalize_features(features, model):
    """
    Normalize different model output formats into:

    cls_features: Tensor [B, D]
    patch_features: Tensor [B, P, D], P a perfect square (e.g. 256 = 16x16)
    """

    if isinstance(features, dict):
        cls_features, patch_features = _normalize_feature_dict(features)
    elif torch.is_tensor(features):
        cls_features, patch_features = _normalize_feature_tensor(features, model)
    else:
        raise TypeError(f"Unsupported feature output type: {type(features)}")

    _assert_square_patch_grid(patch_features)

    return cls_features, patch_features


def _normalize_feature_dict(features):
    """Normalize model outputs returned as dictionaries."""

    if "x_norm_patchtokens" in features:
        patch_features = features["x_norm_patchtokens"]

        if "x_norm_clstoken" in features:
            cls_features = features["x_norm_clstoken"]
        else:
            cls_features = patch_features.mean(dim=1)

        return cls_features, patch_features

    if "last_hidden_state" in features:
        tokens = features["last_hidden_state"]
        cls_features = tokens[:, 0]
        patch_features = tokens[:, 1:]
        return cls_features, patch_features

    raise ValueError(f"Unsupported feature dictionary keys: {features.keys()}")


def _normalize_feature_tensor(features, model):
    """Normalize model outputs returned as tensors."""

    if features.ndim == 3:
        # Shape: [batch, tokens, dim]
        # Layout: [CLS, register_1..register_R, patch_1..patch_P].
        # num_prefix_tokens (CLS + any register tokens) comes from the model
        # itself so its own register-token count is respected.
        num_prefix_tokens = getattr(model, "num_prefix_tokens", 1)
        cls_features = features[:, 0]
        patch_features = features[:, num_prefix_tokens:]
        return cls_features, patch_features

    if features.ndim == 2:
        # Shape: [batch, dim]
        # Model only returned one vector per tile; no patch tokens available.
        raise ValueError(
            "Model returned a single vector per tile (no patch "
            "tokens) — the CLS+patch schema requires patch tokens."
        )

    raise ValueError(f"Unsupported tensor feature shape: {features.shape}")


def _features_to_cpu(cls_features, patch_features):
    """Detach features and move them to CPU for storage."""

    return cls_features.detach().cpu(), patch_features.detach().cpu()


def _metadata_value(batch, key, index):
    """Read one metadata value from a tensor or list-like batch field."""

    value = batch[key]

    if torch.is_tensor(value):
        return value[index].item()

    return value[index]


def _build_tile_metadata(batch, index):
    """Build metadata dictionary for one tile."""

    return {
        "slide": batch["slide"][index],
        "level": _metadata_value(batch, "level", index),
        "x": _metadata_value(batch, "x", index),
        "y": _metadata_value(batch, "y", index),
    }


def _print_tile_debug(
    metadata,
    cls_feature,
    patch_features,
    printed_tiles,
):
    """Print feature debugging information for one tile."""

    print(f"\nTile #{printed_tiles}")
    print(f"file: {metadata['slide']}")
    print(f"level={metadata['level']} | x={metadata['x']} | y={metadata['y']}")
    print("CLS feature shape:", tuple(cls_feature.shape))
    print("CLS feature first 10:", cls_feature[:10])

    num_patches = patch_features.shape[0]
    print("number of patches processed:", num_patches)
    print("patch feature shape:", tuple(patch_features.shape))

    patches_to_print = min(10, num_patches)
    for patch_idx in range(patches_to_print):
        print(
            f"patch #{patch_idx} first 10 features:",
            patch_features[patch_idx, :10],
        )


def _collect_batch_metadata_and_debug_prints(
    batch,
    cls_features_cpu,
    patch_features_cpu,
    metadata_all,
    first_slide,
    printed_tiles,
    max_tiles_to_print,
):
    """
    Store metadata for every tile in a batch.

    Also prints debug info for the first N tiles from the first slide.
    """

    batch_size = cls_features_cpu.shape[0]

    for i in range(batch_size):
        metadata = _build_tile_metadata(batch, i)
        metadata_all.append(metadata)

        if first_slide is None:
            first_slide = metadata["slide"]

        should_print = (
            metadata["slide"] == first_slide
            and printed_tiles < max_tiles_to_print
        )

        if should_print:
            _print_tile_debug(
                metadata=metadata,
                cls_feature=cls_features_cpu[i],
                patch_features=patch_features_cpu[i],
                printed_tiles=printed_tiles,
            )

            printed_tiles += 1

    return first_slide, printed_tiles


def _save_slide_features(slide, cls_chunks, patch_chunks, coordinate_chunks, model_dir, model_encoder, tile_params):
    """
    Concatenate one slide's buffered chunks and write them to a single .pt file.

    `patch_features` is only written when patch chunks were buffered (save_patch_features=True).
    """

    output_path = model_dir / f"{Path(slide).stem}.pt"

    contents = {
        "cls_features": torch.cat(cls_chunks, dim=0),
        "coordinates": torch.tensor(
            [xy for chunk in coordinate_chunks for xy in chunk],
            dtype=torch.long,
        ),
        "model_encoder": model_encoder,
        "tile_params": tile_params,
    }

    if patch_chunks:
        contents["patch_features"] = torch.cat(patch_chunks, dim=0)

    torch.save(contents, output_path)

    print(f"Saved features: {output_path}")

    return output_path


# Main function ***********************************************************************************
def generate_features(
    dataloader,
    transform,
    model,
    device,
    max_tiles_to_print: int = 16,
    output_dir: str | Path = "features",
    model_encoder: str = "unknown",
    save_patch_features: bool = True,
):
    """
    Generate per-tile CLS + patch-token feature vectors from a model.

    This assumes each batch has:
      - batch["image"]: Tensor with shape [B, C, H, W]
      - batch["slide"], batch["x"], batch["y"], batch["level"]: metadata

    The model's raw output is normalized (via `_normalize_features`) into
    one CLS-style vector per tile plus a square grid of spatial patch tokens per tile
    (e.g. 256 = 16x16 for a patch14 ViT on a 224x224 tile), with any register tokens
    stripped using the model's own `num_prefix_tokens`. One `.pt` file is written per
    slide, containing every tile's CLS/patch tokens in the tile's original scan order.

    Arguments:
        dataloader: The DataLoader that returns tile batches.
        transform: The model-specific transform.
        model: The model used to generate features.
        device: The torch device used for inference.
        max_tiles_to_print: Number of tiles to print for debugging.
        output_dir: The directory that receives one subfolder per model encoder.
        model_encoder: The name of the subfolder and the encoder tag stored in each file.
        save_patch_features: If True, save the per-patch tokens next to the CLS vectors. If False,
            save only the CLS vectors (the files are ~250x smaller); no "patch_features" key is written.

    Returns a dict with the saved file paths and the number of tiles processed. Features are
    written slide by slide, so memory use does not grow with the dataset.
    """

    print("Generating features...")

    model = _prepare_model(model, device)

    model_dir = Path(output_dir) / model_encoder
    model_dir.mkdir(parents=True, exist_ok=True)

    # Tiling parameters (level, tile size, stride, filter) are stored in every output file
    tile_params = dataloader.dataset.get_config()

    # Only the current slide is buffered; it is written as soon as the next slide starts
    current_slide = None
    cls_chunks = []
    patch_chunks = []
    coordinate_chunks = []
    num_tiles = 0

    # Slides are written by one background thread so inference does not wait on disk writes
    saver = ThreadPoolExecutor(max_workers=1)
    save_futures = []

    def save_in_background(slide, cls_chunks, patch_chunks, coordinate_chunks):
        # Wait for the previous write first, so at most one finished slide is held in memory
        if save_futures:
            save_futures[-1].result()

        save_futures.append(
            saver.submit(
                _save_slide_features,
                slide, cls_chunks, patch_chunks, coordinate_chunks,
                model_dir, model_encoder, tile_params,
            )
        )

    printed_tiles = 0
    first_slide = None

    print(f"Printing first {max_tiles_to_print} tiles...")

    for batch in tqdm(dataloader):
        images = _move_images_to_device(batch, device)

        raw_features = _run_model(
            images=images,
            transform=transform,
            model=model,
            device=device,
        )

        cls_features, patch_features = _normalize_features(raw_features, model)

        cls_features_cpu, patch_features_cpu = _features_to_cpu(
            cls_features,
            patch_features,
        )

        batch_metadata = []

        first_slide, printed_tiles = _collect_batch_metadata_and_debug_prints(
            batch=batch,
            cls_features_cpu=cls_features_cpu,
            patch_features_cpu=patch_features_cpu,
            metadata_all=batch_metadata,
            first_slide=first_slide,
            printed_tiles=printed_tiles,
            max_tiles_to_print=max_tiles_to_print,
        )

        # Split the batch into runs of consecutive tiles from the same slide
        run_start = 0

        for index in range(1, len(batch_metadata) + 1):
            at_end = index == len(batch_metadata)

            if not at_end and batch_metadata[index]["slide"] == batch_metadata[run_start]["slide"]:
                continue

            slide = batch_metadata[run_start]["slide"]

            if current_slide is not None and slide != current_slide:
                save_in_background(current_slide, cls_chunks, patch_chunks, coordinate_chunks)
                cls_chunks, patch_chunks, coordinate_chunks = [], [], []

            current_slide = slide
            cls_chunks.append(cls_features_cpu[run_start:index])
            if save_patch_features:
                patch_chunks.append(patch_features_cpu[run_start:index])
            coordinate_chunks.append(
                [[m["x"], m["y"]] for m in batch_metadata[run_start:index]]
            )
            num_tiles += index - run_start
            run_start = index

    if current_slide is None:
        saver.shutdown()
        raise ValueError("The dataloader produced no batches.")

    save_in_background(current_slide, cls_chunks, patch_chunks, coordinate_chunks)

    # Wait for every write to finish (re-raises any write error)
    saver.shutdown(wait=True)
    saved_paths = [future.result() for future in save_futures]

    return {
        "saved_paths": saved_paths,
        "num_tiles": num_tiles,
    }
