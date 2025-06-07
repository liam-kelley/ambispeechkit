"""
Computing spherical harmonic coefficients using scipy's sph_harm function.
This script is based on the implementation by Marc1701.
"""

import numpy as np
import numpy.typing as npt
import scipy.special as sp
import torch

from typing import Union
from scipy.linalg import block_diag
from collections import OrderedDict


def sph_harm_coefficients(
    N: int,
    polar_angle: Union[float, npt.ArrayLike],
    azimuth_angle: Union[float, npt.ArrayLike],
    real_sh=True,
) -> npt.ArrayLike:
    """
    Generate spherical harmonic coefficents of a certain order for 1 or more DOA.
    Using scipy's sph_harm function.

    Args:
        N (int): Order of spherical harmonics
        polar_angle (float/array): polar angles (theta in SciPy convention)
        azimuth_angle (float/array): azimuth angles (phi in SciPy convention)
        sh_type (str): Type of spherical harmonics (real or complex)

    Returns:
        numpy.ndarray: Spherical harmonic coefficients
    """
    # Convert inputs to numpy arrays for consistent handling
    polar_array = np.asarray(polar_angle)
    azimuth_array = np.asarray(azimuth_angle)

    # Determine number of queried angles
    if polar_array.ndim == 0:  # Single value (scalar)
        Q = 1
        # Reshape to handle scalar inputs in array operations
        polar_array = polar_array.reshape(1)
        azimuth_array = azimuth_array.reshape(1)
    else:  # Multiple values
        assert (
            polar_array.shape == azimuth_array.shape
        ), "Polar and azimuth angles must have the same shape"
        Q = polar_array.size

    # Ensure arrays are flattened for processing
    polar_angle = polar_array.flatten()
    azimuth_angle = azimuth_array.flatten()

    # Initialize array for sph_harm_coefficients
    sh_coeffs = np.zeros([Q, (N + 1) ** 2], dtype=complex)

    # Compute spherical harmonic for every spherical harmonic index
    for sh_idx in range((N + 1) ** 2):
        # Calculate spherical harmonic degree (n) and spherical harmonic order (m)
        n = int(np.floor(np.sqrt(sh_idx)))  # 0, 1, 1, 1, 2 , ...
        m = sh_idx - (n**2) - n  # 0, -1, 0, 1, -2, ...

        # Compute spherical harmonics
        sh_coeffs[:, sh_idx] = sp.sph_harm(m, n, polar_angle, azimuth_angle).reshape(
            1, -1
        )

    # Convert to real spherical harmonics if requested
    if real_sh:
        sh_coeffs = np.real((_complex_to_real_matrix(N) @ sh_coeffs.T).T)

    sh_coeffs = np.array(sh_coeffs)
    sh_coeffs = np.reshape(sh_coeffs, (Q, -1))

    # Convert to torch
    sh_coeffs = torch.tensor(sh_coeffs, dtype=torch.float32)

    return sh_coeffs


def _complex_to_real_matrix(N: int):
    """
    Returns a complex to real transformation matrix.

    Args:
        N (int): Order of spherical harmonics

    Returns:
        numpy.ndarray: Transformation matrix
    """
    C_dict = OrderedDict()
    for n in range(N + 1):
        C_dict[n] = _complex_to_real_matrix_block(n)
    return block_diag(*[x for _, x in C_dict.items()])


def _complex_to_real_matrix_block(n: int):
    """
    Create a block of the complex to real transformation matrix for a specific sh degree (n).
    Based on the degree (n), it generates a transformation matrix that converts complex spherical harmonics to real spherical harmonics.
    The transformation is based on the sh order (m) and the corresponding indices (mp).
    if m > 0, then we take te real part of the complex harmonic multiplied by sqrt(2)
    if m < 0, then we take the imaginary part of the complex harmonic multiplied by sqrt(2)
    if m = 0, then we take the complex harmonic.
    Or something like this, I'm not sure.

    Args:
        n (int): Spherical harmonic degree

    Returns:
        numpy.ndarray: Transformation matrix
    """
    indices = _rotation_indices(n)
    C_block = np.zeros((2 * n + 1) ** 2, dtype=complex)

    for i, (n, m, mp) in enumerate(indices):
        # Complex to real transformation logic
        if abs(m) != abs(mp):
            C_block[i] = 0
        elif m - mp == 0:
            if m == 0:
                C_block[i] = np.sqrt(2)
            elif m < 0:
                C_block[i] = 1j
            else:
                C_block[i] = int(-1) ** int(m)
        elif m - mp > 0:
            C_block[i] = 1
        elif m - mp < 0:
            C_block[i] = -1j * (int(-1) ** int(m))

    C_block *= 1 / (np.sqrt(2))
    return C_block.reshape(2 * n + 1, 2 * n + 1)


def _rotation_indices(n):
    """
    Generate rotation indices for a given spherical harmonic degree.
    What is a rotation indice ? Maybe this is just the indices of the spherical harmonics ?

    Args:
        n (int): Spherical harmonic degree

    Returns:
        numpy.ndarray: Array of rotation indices
    """
    return np.array([[n, m, mp] for m in range(-n, n + 1) for mp in range(-n, n + 1)])
