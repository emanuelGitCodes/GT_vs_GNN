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


def cuda_build_supports_current_gpu() -> tuple[bool, str]:
    """Return whether the installed PyTorch CUDA build supports this GPU.

    Blackwell GPUs can report as available even when the installed PyTorch
    wheel lacks the matching CUDA architecture. Trying to allocate tensors on
    such a device raises a runtime kernel-image error later in training, so we
    check support before selecting CUDA in auto mode.
    """
    if not torch.cuda.is_available():
        return False, "CUDA is not available."

    major, minor = torch.cuda.get_device_capability(0)
    required_arch = f"sm_{major}{minor}"
    arch_list = set(torch.cuda.get_arch_list())
    if required_arch in arch_list or f"compute_{major}{minor}" in arch_list:
        return True, f"CUDA architecture {required_arch} is supported."

    gpu_name = torch.cuda.get_device_name(0)
    supported = ", ".join(sorted(arch_list)) or "none reported"
    return (
        False,
        f"GPU '{gpu_name}' requires {required_arch}, but this PyTorch build "
        f"supports: {supported}.",
    )


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
            cuda_supported, reason = cuda_build_supports_current_gpu()
            if cuda_supported:
                return torch.device("cuda")
            print(f"[device] CUDA unavailable for this build: {reason}")
            print("[device] Falling back to CPU.")
        return torch.device("cpu")

    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps' is not available.")
        return torch.device("mps")

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' is not available.")
        cuda_supported, reason = cuda_build_supports_current_gpu()
        if not cuda_supported:
            raise RuntimeError(
                "Requested device 'cuda' is not usable with this PyTorch build. "
                f"{reason}"
            )
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
