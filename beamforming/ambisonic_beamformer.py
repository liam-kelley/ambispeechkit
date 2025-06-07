"""
This script implements an Ambisonic Beamformer class that can compute a single channel signal
from ambisonic foa input and directions / covariance matrices using different beamforming techniques.

Throughout this whole script, "channels" will be of (sh_order+1)**2 dimension.
"""

import torch
import torch.nn as nn
import torchaudio.functional as F

from spherical_harmonics.get_max_re_weights import get_max_re_weights_scalor
from spherical_harmonics.get_sph_harm_coeffs import get_sph_harm_coeffs
import numpy as np
from scipy import linalg as la


class Ambisonic_Beamformer(nn.Module):
    """
    Computes a single channel single signal from a direction and a FOA input.

    Can be initialized with these beamformer styles:
    0. omni: Uses the omnidirectional component of the FOA input.
    1. max_di: Uses the maximum directivity index (DI) beamforming weights.
    2. max_re: Uses the maximum response error (RE) beamforming weights.
    3. lc: Uses the beamforming weights from the "Dilated U-Net [...]" paper, which look like LCMV without MV.
    4. mvdr: Uses the minimum variance distortionless response (MVDR) beamforming weights.
    5. souden_mvdr: Uses the Souden MVDR beamforming weights.
    6. lcmv: Uses the linear constrained minimum variance (LCMV) beamforming weights.
    7. max_sisnr: Uses the maximum scale-invariant signal-to-noise ratio (max-SISNR) beamforming weights.
    8. gevd_mwf: Uses the generalized eigenvalue decomposition (GEVD) multichannel weiner filter weights.

    And can have two different sph coefficent implementation styles: cheind, marc1701
    with a specific order of spherical harmonics. (0-8+).
    But only orders 0-8 are supported with autograd via the cheind implementation.

    Forward Args:
        specgram: complex Ambisonic stft input. (batch_size, channels , f_bins, t_bins)
        azimuth: Target azimuthal angle for the beamformer. (batch_size) (Optional)
        elevation: Target elevation (polar) angle for the beamformer. (batch_size) (Optional)
        interference_azimuth: Interference azimuthal angles for the beamformer. (batch_size, num_interferences) (Optional)
        interference_elevation: Interference elevation (polar) angles for the beamformer. (batch_size, num_interferences) (Optional)
        speech_specgram: complex Speech stft input. (batch_size, channels , f_bins, t_bins) (Optional)
        noise_specgram: complex Noise stft input. (batch_size, channels , f_bins, t_bins) (Optional)

    Forward Returns:
        Beamformed outputs. (batch_size, f_bins, t_bins)
    """

    def __init__(
        self,
        bf_style: str = "max_di",  # Options : omni, max_di, max_re, lc, mvdr, souden_mvdr, lcmv, max_sisnr, gevd_mwf
        sph_order: int = 1,  # Spherical harmonics order
        sph_implementation: str = "cheind",  # Options : cheind, marc1701
    ):
        super().__init__()

        self.bf_style = bf_style
        self.sph_order = sph_order
        self.sph_implementation = sph_implementation

        # Precomputed max RE weights scalor, initialized if needed.
        self.max_re_weights_scalor = None
        self.check_init()
        print(
            f"Beamformer initialized with style set to {self.bf_style} and sph implementation {self.sph_implementation}."
        )

    def forward(
        self,
        specgram: torch.Tensor,
        azimuth: torch.Tensor = None,
        elevation: torch.Tensor = None,
        interference_azimuth: torch.Tensor = None,
        interference_elevation: torch.Tensor = None,
        speech_specgram: torch.Tensor = None,
        noise_specgram: torch.Tensor = None,
    ) -> torch.Tensor:
        # Specgram assertions
        for specgram_to_check in [specgram, speech_specgram, noise_specgram]:
            if specgram_to_check is not None:
                self.check_specgram(specgram_to_check)

        # Direction assertions
        for direction_to_check in [
            azimuth,
            elevation,
            interference_azimuth,
            interference_elevation,
        ]:
            if direction_to_check is not None:
                self.check_direction(azimuth)

        weights = self.get_weights(
            specgram,
            azimuth,
            elevation,
            interference_azimuth,
            interference_elevation,
            speech_specgram,
            noise_specgram,
        )  # (batch_size, f_bins, channels)

        # beamformer weights need to be complex floats for consistency with conventional
        # mic arrays, and to be compatible with the apply_beamforming function.
        weights = weights.type(torch.complex64)

        # Apply beamformer weights to specgram
        beamformed_output = F.apply_beamforming(
            beamform_weights=weights, specgram=specgram
        )
        return beamformed_output  # (batch_size, f_bins, t_bins)

    def get_stv_or_rtf_or_sph_harm_coeffs(
        self, azimuth: torch.Tensor, elevation: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the steering vector / relative transfer function / spherical harmonic
        coefficients for the given azimuth and elevation of a target source.
        In ambisonics, this is just the spherical harmonic coefficients evaluated
        in the target direction.

        Args:
            azimuth: Azimuth angles. (batch_size, n_angles)
            elevation: Elevation angles. (batch_size, n_angles)

        Returns:
            Steering vector. (batch_size, n_angles, channels)
        """
        sph_harm_coeffs = get_sph_harm_coeffs(
            azimuth=azimuth,
            elevation=elevation,
            sh_order=self.sph_order,
            implementation=self.sph_implementation,
        )
        return sph_harm_coeffs

    def get_weights(
        self,
        specgram: torch.Tensor,
        azimuth: torch.Tensor = None,
        elevation: torch.Tensor = None,
        interference_azimuth: torch.Tensor = None,
        interference_elevation: torch.Tensor = None,
        speech_specgram: torch.Tensor = None,
        noise_specgram: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Interfaces with the specific beamformer weights getter functions.
        Returns weights (batch_size, f_bins, channels)
        """
        # Get beamformer weights (batch_size, f_bins, channels)
        match self.bf_style:
            case "omni":
                ones_and_zeroes = [1.0] + [0.0] * ((self.sph_order + 1) ** 2 - 1)
                weights = torch.tensor(ones_and_zeroes, device=specgram.device)
                weights = weights.unsqueeze(0).unsqueeze(0)  # (1, 1, channels)
                weights = weights.repeat(specgram.shape[0], specgram.shape[2], 1)
            case "max_di":
                assert azimuth is not None
                assert elevation is not None
                weights = self.get_max_di_weights(azimuth, elevation)
            case "max_re":
                assert azimuth is not None
                assert elevation is not None
                weights = self.get_max_re_weights(azimuth, elevation)
            case "lc":
                assert azimuth is not None
                assert elevation is not None
                assert interference_azimuth is not None
                assert interference_elevation is not None
                weights = self.get_lc_weights(
                    azimuth, elevation, interference_azimuth, interference_elevation
                )
            case "mvdr":
                assert azimuth is not None
                assert elevation is not None
                assert noise_specgram is not None
                psd_or_cov_n = F.psd(noise_specgram)
                weights = self.get_mvdr_weights(azimuth, elevation, psd_or_cov_n)
            case "souden_mvdr":
                assert noise_specgram is not None
                psd_or_cov_s = F.psd(
                    speech_specgram if speech_specgram is not None else specgram
                )
                psd_or_cov_n = F.psd(noise_specgram)
                weights = self.get_souden_mvdr_weights(psd_or_cov_s, psd_or_cov_n)
            case "lcmv":
                assert azimuth is not None
                assert elevation is not None
                assert noise_specgram is not None
                psd_or_cov_n = F.psd(noise_specgram)
                weights = self.get_lcmv_weights(
                    azimuth,
                    elevation,
                    interference_azimuth,
                    interference_elevation,
                    psd_or_cov_n,
                )
            case "max_sisnr":
                assert noise_specgram is not None
                psd_or_cov_s = F.psd(
                    speech_specgram if speech_specgram is not None else specgram
                )
                psd_or_cov_n = F.psd(noise_specgram)
                weights = self.get_max_sisnr_weights(psd_or_cov_s, psd_or_cov_n)
            case "gevd_mwf":
                assert noise_specgram is not None
                psd_or_cov_s = F.psd(
                    speech_specgram if speech_specgram is not None else specgram
                )
                psd_or_cov_n = F.psd(noise_specgram)
                weights = self.get_gevd_mwf_weights(psd_or_cov_s, psd_or_cov_n)
            case _:
                raise ValueError("Invalid beamformer style.")

        self.check_beamformer_weights(weights)

        return weights  # (batch_size, f_bins, channels)

    def get_max_di_weights(self, azimuth, elevation):
        """
        Calculates the maximum directivity index (DI) weights for the given azimuth and elevation.
        The max DI weights are directly given from the steering vector / spherical harmonic coefficients.

        Args:
            azimuth: Azimuth angles. (batch_size)
            elevation: Elevation angles. (batch_size)

        Returns:
            Maximum DI weights. (batch_size, f_bins , channels)
        """
        sph_harm_coeffs = self.get_stv_or_rtf_or_sph_harm_coeffs(azimuth, elevation)
        # add f_bins dimension (Full band beamforming)
        weights = sph_harm_coeffs.unsqueeze(1)
        return weights

    def get_max_re_weights(self, azimuth, elevation):
        """
        Calculates the maximum response error (RE) weights for the given azimuth and elevation.
        The max RE weights are the steering vector / spherical harmonic coefficients multiplied by a
        constant scaling factor.

        Args:
            azimuth: Azimuth angles. (batch_size)
            elevation: Elevation angles. (batch_size)

        Returns:
            Maximum RE weights. (batch_size, f_bins , channels)
        """
        # get steering vector
        sph_harm_coeffs = self.get_stv_or_rtf_or_sph_harm_coeffs(azimuth, elevation)

        # add f_bins dimension (Full band beamforming)
        weights = sph_harm_coeffs.unsqueeze(1)

        # Get the precomputed max RE weights scalor if not already done
        if self.max_re_weights_scalor is None:
            self.max_re_weights_scalor = (
                get_max_re_weights_scalor(
                    sph_order=self.sph_order, device=weights.device
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

        weights = weights * self.max_re_weights_scalor
        return weights

    def get_lc_weights(
        self, speech_azimuth, speech_elevation, noise_azimuths, noise_elevations
    ):
        """
        Calculates the lc beamformer weights using the pseudo-inverse of the concatenated steering vectors.
        This implementation is from the "Dilated U-net [...]" paper.
        TODO Check this implementation.

        Args:
            speech_azimuth: Azimuth of the speech source. (batch_size)
            speech_elevation: Elevation of the speech source. (batch_size)
            noise_azimuths: Azimuths of the noise sources. (batch_size, num_noises)
            noise_elevations: Elevations of the noise sources. (batch_size, num_noises)

        Returns:
            lc beamformer weights. (batch_size, f_bins, channels)
        """
        # Ensure noise azimuths and elevations have a num_noises dimension
        if noise_azimuths.dim() == 1:
            noise_azimuths = noise_azimuths.unsqueeze(1)
        if noise_elevations.dim() == 1:
            noise_elevations = noise_elevations.unsqueeze(1)

        # Get the sph harm coeffs
        speech_steering_vector = self.get_stv_or_rtf_or_sph_harm_coeffs(
            speech_azimuth, speech_elevation
        )  # (batch_size, channels)
        noise_steering_vectors = [
            self.get_stv_or_rtf_or_sph_harm_coeffs(
                noise_azimuths[:, i], noise_elevations[:, i]
            )
            for i in range(noise_azimuths.shape[1])
        ]  # List of (batch_size, channels)

        # Concatenate the speech and noise steering vectors (batch_size, channels, num_sources)
        # TODO check if this is the correct concatenation
        steering_matrix = torch.stack(
            [speech_steering_vector] + noise_steering_vectors, dim=-1
        )

        # Compute the pseudo-inverse of the steering matrix (batch_size, num_sources, channels)
        pseudo_inverse = torch.linalg.pinv(steering_matrix)

        # Select the first column of the pseudo-inverse (corresponding to the speech source)
        # TODO check if this is the correct column to select
        weights = pseudo_inverse[:, 0, :]  # (batch_size, channels)

        # Add the frequency bins dimension (Full band beamforming)
        weights = weights.unsqueeze(1)  # (batch_size, f_bins, channels)

        return weights

    def get_mvdr_weights(self, azimuth, elevation, psd_or_cov_n, ref_mode=None):
        """
        Calculates the MVDR weights for the given azimuth, elevation, and the noise covariance matrix.
        The reference mode is used to determine the reference vector.
        Using the torchaudio implementation of MVDR.

        Args:
            speech_azimuth: Azimuth of the speech source. (batch_size)
            speech_elevation: Elevation of the speech source. (batch_size)
            psd_or_cov_n: Noise Power Spectral Density / Covariance matrix. (batch_size, f_bins, channels, channels)

        Returns:
            MVDR beamformer weights. (batch_size, f_bins, channels)
        """

        # get steering vector
        sph_harm_coeffs = self.get_stv_or_rtf_or_sph_harm_coeffs(azimuth, elevation)
        if ref_mode == "w":  # Using the W channel as reference
            reference_vector = 0
        else:
            reference_vector = None

        # expand frequency bins dimension
        sph_harm_coeffs = sph_harm_coeffs.unsqueeze(1).repeat(
            1, psd_or_cov_n.shape[1], 1
        )  # (batch_size, f_bins, channels)

        # TODO Ideally, avoid switching to complex64 here...
        sph_harm_coeffs = sph_harm_coeffs.type(torch.complex64)
        psd_or_cov_n = psd_or_cov_n.type(torch.complex64)

        weights = F.mvdr_weights_rtf(sph_harm_coeffs, psd_or_cov_n, reference_vector)
        return weights

    def get_souden_mvdr_weights(self, psd_or_cov_s, psd_or_cov_n):
        """
        Calculates the souden MVDR weights for the given speech and noise covariance matrix.
        The reference mode is used to determine the reference vector.
        Using the torchaudio implementation of MVDR.

        Args:
            psd_or_cov_s: Speech Power Spectral Density / Covariance matrix. (batch_size, f_bins, channels, channels)
            psd_or_cov_n: Noise Power Spectral Density / Covariance matrix. (batch_size, f_bins, channels, channels)

        Returns:
            Souden MVDR beamformer weights. (batch_size, f_bins, channels)
        """
        reference_vector = 0  # Using the W channel as reference # TODO there should be a better way to do this

        # TODO Ideally, avoid switching to complex64 here...
        psd_or_cov_s = psd_or_cov_s.type(torch.complex64)
        psd_or_cov_n = psd_or_cov_n.type(torch.complex64)

        weights = F.mvdr_weights_souden(psd_or_cov_s, psd_or_cov_n, reference_vector)
        return weights

    def get_lcmv_weights(
        self,
        speech_azimuth,
        speech_elevation,
        noise_azimuths,
        noise_elevations,
        psd_or_cov_n,
        diag_loading=False,
    ):
        """
        Calculates the LCMV weights for the given azimuth, elevation, and the noise covariance matrix.
        Implementation from Diego Di Carlo, slightly refactored for torch.

        Args:
            speech_azimuth: Azimuth of the speech source. (batch_size, 1)
            speech_elevation: Elevation of the speech source. (batch_size, 1)
            noise_azimuths: Azimuths of the noise sources. (batch_size, num_noises)
            noise_elevations: Elevations of the noise sources. (batch_size, num_noises)
            psd_or_cov_n: Noise Power Spectral Density / Covariance matrix. (batch_size, f_bins, channels, channels)

        Returns:
            lcmv beamformer weights. (batch_size, f_bins, channels)
        """
        # Ensure noise azimuths and elevations have a num_noises dimension
        if noise_azimuths.dim() == 1:
            noise_azimuths = noise_azimuths.unsqueeze(1)
        if noise_elevations.dim() == 1:
            noise_elevations = noise_elevations.unsqueeze(1)

        # Get the sph harm coeffs
        speech_steering_vector = self.get_stv_or_rtf_or_sph_harm_coeffs(
            speech_azimuth, speech_elevation
        )  # (batch_size, channels)
        noise_steering_vectors = [
            self.get_stv_or_rtf_or_sph_harm_coeffs(
                noise_azimuths[:, i], noise_elevations[:, i]
            )
            for i in range(noise_azimuths.shape[1])
        ]  # List of (batch_size, channels)

        # (batch_size, f_bins, channels, num_sources)
        A = torch.stack(
            [speech_steering_vector] + noise_steering_vectors, dim=-1
        ).unsqueeze(1)
        q = torch.tensor(
            [1] + [0] * len(noise_steering_vectors),
            dtype=torch.float32,
            device=psd_or_cov_n.device,
        )  # Desired constraints

        # Compute weights with matrix constraints
        reg = 0.0 if diag_loading else 1e-7
        eye_matrix = torch.eye(psd_or_cov_n.shape[-1], device=psd_or_cov_n.device)
        invRn = torch.linalg.inv(psd_or_cov_n + reg * eye_matrix)
        invRn = torch.real(invRn).double()
        A = torch.real(A).double()
        invRn_A = torch.einsum("bfij,bfjk->bfik", invRn, A)
        AH_invRn_A = torch.einsum("bfik,bfiK->bfkK", A.conj(), invRn_A)
        inv_AH_invRn_A = torch.linalg.inv(
            AH_invRn_A + reg * torch.eye(AH_invRn_A.shape[-1], device=AH_invRn_A.device)
        )
        w = torch.einsum("bfik,bfkK->bfiK", invRn_A, inv_AH_invRn_A)
        w = torch.einsum("bfiK,k->bfi", w, q)

        # # Verify constraints
        # assert np.allclose(
        #     np.einsum("fi,fi->f", w.conj(), speech_steering_vector),
        #     np.ones(w.shape[0]) + 1j * 0,
        # )
        # assert np.allclose(
        #     np.einsum("fi,fi->f", w.conj(), noise_steering_vectors),
        #     np.zeros(w.shape[0]) + 1j * 0,
        # )
        return w

    def get_max_sisnr_weights(self, psd_or_cov_s, psd_or_cov_n):
        """
        Maximum SINR (Signal-to-Interference-plus-Noise Ratio) Beamforming
        Implementation from Diego Di Carlo, slightly refactored for torch,
        but since we aren't going to be backpropagating through this, we can just use the
        scipy implementation of eigh.

        Parameters:
        - psd_or_cov_s: Signal covariance matrix (batch_size, f_bins, channels, channels).
        - psd_or_cov_n: Noise covariance matrix (batch_size, f_bins, channels, channels).

        Returns:
        - Weights that maximize signal quality across channels (batch_size, f_bins, channels).
        """
        device = psd_or_cov_s.device  # Get the device of the input tensors

        # Check for NaNs in input tensors
        if torch.isnan(psd_or_cov_s).any() or torch.isnan(psd_or_cov_n).any():
            print("Warning: NaN values detected in input covariance matrices")
            # Replace NaNs with zeros
            psd_or_cov_s = torch.nan_to_num(psd_or_cov_s, nan=0.0)
            psd_or_cov_n = torch.nan_to_num(psd_or_cov_n, nan=0.0)

        psd_or_cov_s = (
            psd_or_cov_s.detach().cpu().numpy()
        )  # Detach and move to CPU for scipy
        psd_or_cov_n = (
            psd_or_cov_n.detach().cpu().numpy()
        )  # Detach and move to CPU for scipy

        # Number of channels
        n_channels = psd_or_cov_s.shape[-1]

        # Flatten batch_size and f_bins dimensions for easier processing
        batch_size, f_bins = psd_or_cov_s.shape[:2]
        psd_or_cov_s_flat = psd_or_cov_s.reshape(-1, n_channels, n_channels)
        psd_or_cov_n_flat = psd_or_cov_n.reshape(-1, n_channels, n_channels)

        # Solve generalized eigenvalue problem
        # Select the eigenvector corresponding to the largest eigenvalue
        weights_flat = [
            la.eigh(
                a=rs,  # Signal covariance
                b=rn
                + np.eye(rn.shape[0])
                * 1e-6,  # Noise covariance, with small regularization term to guarantee positive definite.
                # eigvals=(n_channels - 1, n_channels - 1),  # Last eigenvalue/vector
            )[1][0]
            for rs, rn in zip(psd_or_cov_s_flat, psd_or_cov_n_flat)
        ]
        weights_flat = np.array(weights_flat)

        # Reshape back to original dimensions
        weights = weights_flat.reshape(batch_size, f_bins, n_channels)
        weights = torch.tensor(
            weights, dtype=torch.complex64, device=device
        )  # (batch_size, f_bins, channels)

        # Final NaN check on output weights
        if torch.isnan(weights).any():
            print("Warning: NaN values in final weights, replacing with zeros")
            weights = torch.nan_to_num(weights, nan=0.0)

        return weights

    def get_gevd_mwf_weights(self, psd_or_cov_s, psd_or_cov_n):
        """
        Calculates the w rank-1 Multichannel Wiener Filter (wGEVD-MWF) weights using generalized eigenvalue decomposition.
        This implementation is from "MULTICHANNEL SPEECH SEPARATION WITH RECURRENT NEURAL NETWORKS
        FROM HIGH-ORDER AMBISONICS RECORDINGS". And might be totally wrong. TODO check this implementation.

        Args:
            speech_covariance: Speech covariance matrix. (batch_size, f_bins, channels (4), channels (4))
            noise_covariance: Noise covariance matrix. (batch_size, f_bins, channels (4), channels (4))

        Returns:
            gevd mwf weights beamformer weights. (batch_size, f_bins, channels)
        """

        # Compute weights with matrix constraints
        reg = 1e-7
        reg = reg * torch.eye(psd_or_cov_n.shape[-1], device=psd_or_cov_n.device)
        inv_noise_covariance = torch.linalg.inv(psd_or_cov_n + reg)
        inv_noise_covariance_speech_covariance = torch.einsum(
            "bfij,bfjk->bfik", inv_noise_covariance, psd_or_cov_s
        )

        # # Get speech covariance matrix rank-1 approximation
        eigenvalues, eigenvectors = torch.linalg.eigh(
            inv_noise_covariance_speech_covariance + reg
        )

        # Get the largest eigenvalue (last one) and corresponding eigenvector
        largest_eigvalue = eigenvalues[..., -1]  # (batch_size, f_bins)
        associated_eigenvector = eigenvectors[..., -1]  # (batch_size, f_bins, channels)

        speech_covariance_rank1_approx = torch.einsum(
            "bf,bfi,bfj->bfij",
            largest_eigvalue,  # (batch_size, f_bins)
            associated_eigenvector,  # (batch_size, f_bins, channels)
            associated_eigenvector.conj(),  # (batch_size, f_bins, channels)
        )  # (batch_size, f_bins, channels, channels)

        # Weight computation
        inv_term = torch.linalg.inv(speech_covariance_rank1_approx + psd_or_cov_n + reg)
        weights = torch.einsum(
            "bfij,bfjk->bfik", inv_term, speech_covariance_rank1_approx
        )  # (batch_size, f_bins, channels, channels)

        # Select first column of the weights
        weights = weights[:, :, :, 0]  # (batch_size, f_bins, channels)

        return weights

    def check_init(self):
        assert self.bf_style in [
            "omni",
            "max_di",
            "max_re",
            "lc",
            "mvdr",
            "souden_mvdr",
            "lcmv",
            "max_sisnr",
            "gevd_mwf",
        ], "Invalid beamformer style."

        assert self.sph_order >= 0, "Spherical harmonics order must be non-negative."

        assert (
            self.sph_order <= 8
        ), "Spherical harmonics order greater than 8 are not supported."

        assert self.sph_implementation in [
            "cheind",
            "marc1701",
        ], "Invalid spherical harmonics implementation. Use 'cheind' or 'marc1701'."

    def check_specgram(self, specgram):
        """
        Asserts that the input specgram is valid.

        Args:
            specgram: Input tensor. (batch_size, channels, height, width)

        Raises:
            ValueError: If the input shape is not correct, or if the input contains NaN values.
        """
        # Shape checks
        if len(specgram.shape) != 4:
            raise ValueError(
                f"Expected input shape of 4 dimensions (batch_size, channels, height, width), got {len(specgram.shape)} dimensions."
            )
        if specgram.shape[1] != (self.sph_order + 1) ** 2:
            raise ValueError(
                f"Expected input channels to be {(self.sph_order+ 1)**2}: ({self.sph_order} order Ambisonics), got {specgram.shape[1]} channels."
            )

        # NaN checks
        if torch.isnan(specgram).any():
            print(
                f"Warning: NaN values detected in input specgram of the {self.bf_style} beamformer"
            )
            print(specgram[0])
            raise ValueError("Input specgram contains NaN values")

        # Check is complex
        if not specgram.is_complex():
            raise TypeError(f"Expected spectrogram to be complex.")

    def check_direction(self, direction):
        """
        Asserts that the input direction is valid.

        Args:
            direction: Input tensor. (batch_size) or (batch_size, n_angles)
        Raises:
            ValueError: If the input shape is not correct, or if the input contains NaN values.
        """
        # Shape checks
        if len(direction.shape) not in [1, 2]:
            raise ValueError(
                f"Expected input shape of 1 or 2 dimensions (batch_size) or (batch_size, n_angles), got {direction.shape}"
            )
        # Perform NaN checks on the input direction
        if torch.isnan(direction).any():
            print("Warning: NaN values detected in input direction")
            raise ValueError("Input direction contains NaN values")

    def check_beamformer_weights(self, weights):
        """
        Asserts that the beamformer weights are valid.

        Args:
            weights: Beamformer weights. (batch_size, f_bins, channels)

        Raises:
            ValueError: If the beamformer weights are not valid.
        """
        if len(weights.shape) != 3:
            print(weights.shape)
            print(weights)
            raise ValueError(
                f"Expected beamformer weights shape of 3 dimensions (batch_size, f_bins, channels), got {len(weights.shape)} dimensions."
            )
        if weights.shape[2] != (self.sph_order + 1) ** 2:
            print(weights.shape)
            print(weights)
            raise ValueError(
                f"Expected beamformer weights channels to be {(self.sph_order + 1) ** 2}, got {weights.shape[2]} channels."
            )

        # TODO : understand these contraints

        # # Verify beamforming constraint
        # assert np.allclose(
        #     np.einsum("fi,fi->f", a1.conj(), w), np.ones(w.shape[0]) + 1j * 0
        # )

        # # LCMV Verify constraints
        # assert np.allclose(
        # np.einsum("fi,fi->f", w.conj(), a1_good), np.ones(w.shape[0]) + 1j * 0
        # )
        # assert np.allclose(
        #     np.einsum("fi,fi->f", w.conj(), a1_bad), np.zeros(w.shape[0]) + 1j * 0
        # )
