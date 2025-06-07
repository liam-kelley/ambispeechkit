import torch

from ambispeechkit.spherical_harmonics.get_sph_harm_coeffs import (
    get_sph_harm_coeffs,
)


def upsample_mono_signal(
    signal_spec: torch.Tensor,
    signal_azimuth: torch.Tensor,
    signal_elevation: torch.Tensor,
    sh_order: int = 1,
    sh_implementation: str = "cheind",
):
    """
    Upsample mono signal to multichannel ambisonic using basic spherical harmonics.

    args:
        signal_spec: (batch_size, f_bins, t_bins) - mono signal spectrogram
        signal_azimuth: (batch_size) - azimuth of the signal source in degrees
        signal_elevation: (batch_size) - elevation of the signal source in degrees
        n_channels: int - number of channels in the ambisonic signal
    """
    signal_spec_temp = signal_spec.unsqueeze(1)  # (batch_size, 1, f_bins, t_bins)

    sph_harm_coeffs = (
        get_sph_harm_coeffs(
            azimuth=signal_azimuth,
            elevation=signal_elevation,
            sh_order=sh_order,
            implementation=sh_implementation,
        )
        .unsqueeze(-1)
        .unsqueeze(-1)
    )  # (batch_size, n_channels, 1, 1)

    upsampled_signal_spec = (
        signal_spec_temp * sph_harm_coeffs
    )  # (batch_size, n_channels, f_bins, t_bins)

    amb_signal_specgram = upsampled_signal_spec.type(torch.cfloat)

    return amb_signal_specgram
