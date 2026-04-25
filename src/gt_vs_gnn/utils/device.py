"""Device selection utility with optional backend override.

Usage
-----
    from gt_vs_gnn.utils.device import get_device, sanity_check

    device = get_device("auto")
    sanity_check(device)
"""

from __future__ import annotations

from typing import Literal

import torch


DevicePreference = Literal["auto", "mps", "cuda", "cpu"]


def get_device(preference: DevicePreference = "auto") -> torch.device:
    """Return the selected device with optional manual override.

    Parameters
    ----------
    preference:
        One of ``{"auto", "mps", "cuda", "cpu"}``.
        - ``auto``: MPS → CUDA → CPU fallback priority.
        - ``mps`` / ``cuda`` / ``cpu``: force a specific backend.

    Raises
    ------
    RuntimeError
        If a forced backend is requested but not available.
    """

    if preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps' is not available.")
        return torch.device("mps")

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' is not available.")
        return torch.device("cuda")

    if preference == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device preference: {preference}")


def _current_memory_mb(device: torch.device) -> float:
    """Return allocated memory on the requested device in MB."""
    if device.type == "mps":
        return float(torch.mps.current_allocated_memory() / 1e6)
    if device.type == "cuda":
        return float(torch.cuda.memory_allocated(device=device) / 1e6)
    return 0.0


def _memory_info(device: torch.device) -> str:
    """Return a short allocated-memory info string for logs."""
    if device.type in {"mps", "cuda"}:
        return (
            f"  |  {device.type.upper()} allocated: {_current_memory_mb(device):.2f} MB"
        )
    return ""


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

    mem_info = _memory_info(device)

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
    _device = get_device("auto")
    sanity_check(_device)
