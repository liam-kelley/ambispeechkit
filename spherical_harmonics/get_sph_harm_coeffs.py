"""
This script interfaces with spherical harmonics implementations to compute
spherical harmonic coefficients based on azimuth and elevation angles.
"""

import spherical_harmonics.real_sph_harm_coeffs_by_cheind as cheind
from spherical_harmonics.sph_harm_coeffs_by_marc1701 import sph_harm_coefficients
from spherical_harmonics.sph_utils import azimuth_elevation_to_cartesian

import torch


def get_sph_harm_coeffs(
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    sh_order: int = 1,
    implementation: str = "cheind",
):
    """
    Get spherical harmonics using the scripts from cheind or marc1701.
    The implementation from cheind is precomputed. The implementation from marc1701 is
    computed on the fly.

    "The steering vector describes how a source at a particular direction contributes to
    the signals received by an array or the components of a spatial representation.
    In Higher-Order Ambisonics (HOA), the steering vector maps a source direction to its
    spherical harmonic components." spherical harmonics are our equivalent to the steering
    vector.

    Args:
        azimuth (torch.Tensor): Azimuth angles in radians. (batch_size) or (batch_size, n_angles).
        elevation (torch.Tensor): Elevation angles in radians. (batch_size) or (batch_size, n_angles).
        sh_order (int): Order of spherical harmonics.
        implementation (str): Implementation to use ("cheind" or "marc1701").

    Returns:
        torch.Tensor: Spherical harmonics coefficients.
            (batch_size, channels (sh_order+1 ** 2)) or (batch_size, n_angles, channels (sh_order+1 ** 2)).
    """
    assert (
        azimuth.shape == elevation.shape
    ), "Azimuth and elevation tensors must have the same shape."
    assert (
        azimuth.ndim == 1 or azimuth.ndim == 2
    ), "Azimuth and elevation tensors must be 1D or 2D."

    if implementation == "cheind":
        xyz = azimuth_elevation_to_cartesian(azimuth, elevation)

        match sh_order:
            case 0:
                sph_harm_coeffs = cheind.rsh_cart_0(xyz)
            case 1:
                sph_harm_coeffs = cheind.rsh_cart_1(xyz)
            case 2:
                sph_harm_coeffs = cheind.rsh_cart_2(xyz)
            case 3:
                sph_harm_coeffs = cheind.rsh_cart_3(xyz)
            case 4:
                sph_harm_coeffs = cheind.rsh_cart_4(xyz)
            case 5:
                sph_harm_coeffs = cheind.rsh_cart_5(xyz)
            case 6:
                sph_harm_coeffs = cheind.rsh_cart_6(xyz)
            case 7:
                sph_harm_coeffs = cheind.rsh_cart_7(xyz)
            case 8:
                sph_harm_coeffs = cheind.rsh_cart_8(xyz)
            case _:
                raise ValueError(
                    "Unsupported spherical harmonics order. Use marc1701 implementation's instead."
                )

    elif implementation == "marc1701":
        sph_harm_coeffs = sph_harm_coefficients(
            N=sh_order, polar_angle=elevation, azimuth_angle=azimuth
        )
    else:
        raise ValueError("Invalid implementation specified.")

    return sph_harm_coeffs
