# IMPORTS *****************************************************************************************
# External libraries
import torch


# Main function ***********************************************************************************
def get_device(preferred: str | None = None) -> torch.device:
    """
    Pick the torch device to run on: CUDA, then Apple MPS, then CPU.

    * Arguments:
    preferred: Optional device type ("cuda", "mps" or "cpu") to force, e.g. to run on
        the CPU of a machine that has a GPU. Raises if that device is not available.
    """

    # STEP 1: Check which accelerators are available
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }

    # STEP 2: Honor an explicit request, failing clearly when it cannot be met
    if preferred is not None:
        if preferred not in available:
            raise ValueError(f"Unknown device '{preferred}'; choose from {list(available)}")

        if not available[preferred]:
            raise RuntimeError(f"Requested device '{preferred}' is not available")

        return torch.device(preferred)

    # STEP 3: Otherwise use the first available device in priority order
    for name, is_available in available.items():
        if is_available:
            return torch.device(name)
