"""Device selection utility: MPS → CUDA → CPU priority order.

Usage
-----
    from utils.device import get_device, sanity_check

    device = get_device()
    sanity_check(device)
"""

import torch


def get_device() -> torch.device:
    """Return the best available device: MPS → CUDA → CPU.

    Never hardcode 'mps' or 'cuda' elsewhere in the codebase — always call
    this function so the code runs on any machine.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sanity_check(device: torch.device) -> None:
    """Run a small matmul on *device* to confirm it is operational.

    Also prints allocated memory when running on MPS so we have a baseline
    before any model is loaded.

    Raises
    ------
    AssertionError
        If the output tensor has an unexpected shape.
    """
    x = torch.randn(64, 64, device=device)
    y = torch.matmul(x, x.T)
    assert y.shape == (64, 64), f"Unexpected output shape: {y.shape}"

    mem_info = ""
    if device.type == "mps":
        allocated_mb = torch.mps.current_allocated_memory() / 1e6
        mem_info = f"  |  MPS allocated: {allocated_mb:.2f} MB"

    print(f"[device] Backend: {device}{mem_info}  |  Sanity matmul OK ✓")


def empty_cache(device: torch.device) -> None:
    """Free unused memory on the active device.

    Call this after evaluation loops to avoid MPS memory fragmentation.
    """
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    _device = get_device()
    sanity_check(_device)
