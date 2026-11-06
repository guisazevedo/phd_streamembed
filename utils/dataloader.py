# IMPORTS *****************************************************************************************
# External libraries
import os
import torch
from torch.utils.data import DataLoader, Dataset

# Internal imports ********************************************************************************

# Each DataLoader worker already runs in its own process; one libvips thread each avoids oversubscription
os.environ.setdefault("VIPS_CONCURRENCY", "1")

# Main function ***********************************************************************************
def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    drop_last: bool = False,
    device: torch.device | None = None,
    num_workers: int = 4,
    prefetch_factor: int = 4,
) -> DataLoader:
    """
    Create a DataLoader that reads tiles in worker processes, with CUDA-friendly transfer settings.

    Batches are returned in order, so with shuffle=False the tiles of a slide still arrive
    contiguously (generate_features relies on this).

    This does not move data to CUDA. It only enables pinned CPU memory when
    the device is CUDA, so later `.to(device, non_blocking=True)` calls are faster.

    * Arguments:
    dataset: The dataset used to create batches
    batch_size: The number of samples per batch
    shuffle: Whether to shuffle samples
    drop_last: Whether to drop the final incomplete batch
    device: The torch device used later for inference (pinned memory is enabled only for CUDA)
    num_workers: The number of worker processes that read tiles (0 = load in the main process)
    prefetch_factor: The number of batches each worker prepares ahead (ignored when num_workers is 0)
    """

    # STEP 1: Create the DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=device is not None and device.type == "cuda",
        drop_last=drop_last,
    )