"""
This file contains utility functions for stft processing in the context of
the "Dilated U-net [...]" paper.
"""

import torch


def compute_magnitude(stft: torch.Tensor) -> torch.Tensor:
    """Compute magnitude
    Args:
        stft (torch.Tensor): STFT of the signal.
    Returns:

        torch.Tensor: Magnitude of the STFT.
    """
    # assert complex
    assert stft.is_complex(), "STFT must be complex."
    real = stft.real
    imag = stft.imag
    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7))


def stft(x: torch.Tensor) -> torch.Tensor:
    """Compute STFT with parameters from the "Dilated U-net [...]" paper.
    Args:
        x (torch.Tensor): Time domain Input signal.
    Returns:
        torch.Tensor: STFT of the signal.
    """
    # assert complex
    assert not x.is_complex(), "Input signal must be real."
    # Reshape to (batch, time) for stft
    og_x_shape = x.shape
    x = x.view(-1, x.shape[-1])

    # Stft params from the "Dilated U-net [...]" paper
    window = torch.sin(torch.linspace(0, torch.pi, 1024, device=x.device))
    stft_args = {
        "n_fft": 1024,
        "hop_length": 512,
        "win_length": 1024,
        "window": window,
        "return_complex": True,
    }
    stft = torch.stft(x, **stft_args)
    # Reshape back to original shape
    stft = stft.view(*og_x_shape[:-1], stft.shape[-2], stft.shape[-1])
    # drop last frequency bin to have 512 bins
    # (..., f_bins 513, t_bins) -> (..., f_bins 512, t_bins)
    stft = stft[..., :-1, :]
    return stft


def istft(stft: torch.Tensor) -> torch.Tensor:
    """Compute inverse STFT with parameters from the "Dilated U-net [...]" paper.
    Args:
        stft (torch.Tensor): STFT of the signal.
    Returns:
        torch.Tensor: Inverse STFT of the signal.
    """
    # assert complex
    assert stft.is_complex(), "STFT must be complex."
    # Add one frequency bin of zeros to the end of the tensor to make it 513 bins
    # (..., f_bins 512, t_bins) -> (..., f_bins 513, t_bins)
    xtra_freq_bin = torch.zeros_like(stft[..., 0, :], device=stft.device).unsqueeze(1)
    stft = torch.cat([stft[..., :, :], xtra_freq_bin], dim=-2)

    # Inverse stft from the "Dilated U-net [...]" paper
    window = torch.sin(torch.linspace(0, torch.pi, 1024, device=stft.device))
    istft_args = {
        "n_fft": 1024,
        "hop_length": 512,
        "win_length": 1024,
        "window": window,
        "return_complex": False,
        "onesided": True,
    }
    return torch.istft(stft, **istft_args)


def band_level_normalize(x):
    return x / torch.amax(x + 1e-8, dim=-2, keepdim=True)


def normalize_spectrogram(x):
    return x / torch.mean(compute_magnitude(x), dim=(-1, -2), keepdim=True)


def normalize_magnitude_spectrogram(x):
    return x / torch.mean(x, dim=(-1, -2), keepdim=True)
