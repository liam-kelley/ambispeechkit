"""Implements the Dilated U-net model from the paper:

"A Dilated U-Net based approach for Multichannel Speech Enhancement from First-Order Ambisonics Recordings"
by A. Bosca, A. Guérin, L. Perotin and S. Kitić. https://ieeexplore.ieee.org/document/9287478
"""

import torch
import torch.nn as nn

from ambispeechkit.tools.stft_utils import compute_magnitude, band_level_normalize
from ambispeechkit.beamforming.ambisonic_beamformer import Ambisonic_Beamformer


class Dilated_Unet(nn.Module):
    """
    Estimates masks from the beamformer outputs and the omnidirectional componrnt of the FOA input,
    and applies them to the FOA input's omnidirectional component.

    Can be initialized in two modes:
    1. Normal mode: Uses normal convolutions.
    2. Dilated mode: Uses dilated convolutions for better receptive field.

    Forward Args:
        speech_beamformer_output: Output from the speech DOA beamformer. (batch_size, f_bins, t_bins)
        noise_beamformer_output: Output from the noise DOA beamformer. (batch_size, f_bins, t_bins)
        omni_input: Omnidirectional (W) of Ambisonics input. (batch_size, f_bins, t_bins)

    Forward Returns:
        Speech mask: To be uniformly applied to the FOA signal. (batch_size, f_bins, t_bins)
            We define the Noise Mask as Noise Mask = 1 - Speech Mask.
    """

    def __init__(self, dilated_mode=False):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.max_pool_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.dilated_mode = dilated_mode

        # # Encoder blocks
        in_channels = 3  # Input
        dilation_rate = 1  # Start with no dilation
        for i in range(5):
            out_channels = 16 * (2**i)
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    (
                        nn.Conv2d(  # Dilated convolution if dilated_mode is True
                            out_channels,
                            out_channels,
                            kernel_size=(3, 3),
                            padding=(dilation_rate, 1),
                            dilation=(dilation_rate, 1),
                        )
                        if self.dilated_mode
                        else nn.Conv2d(  # Normal convolution otherwise
                            out_channels, out_channels, kernel_size=(3, 3), padding=1
                        )
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(0.05),  # 5% dropout after each encoder block
                )
            )
            # Separate max pooling layer
            self.max_pool_layers.append(
                nn.MaxPool2d(kernel_size=(2, 1)) if i < 4 else nn.Identity()
            )
            in_channels = out_channels
            dilation_rate *= 2  # Double the dilation rate for the next block

        # # Decoder blocks
        dilation_rate //= 2  # Start with the maximum dilation rate from the encoder
        for i in reversed(range(4)):
            out_channels = 16 * (2**i)
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels, out_channels, kernel_size=(2, 1), stride=(2, 1)
                    ),
                    (
                        nn.Conv2d(  # Dilated convolution if dilated_mode is True
                            out_channels * 2,  # From Skip connection
                            out_channels,
                            kernel_size=(3, 3),
                            padding=(dilation_rate, 1),
                            dilation=(dilation_rate, 1),
                        )
                        if self.dilated_mode
                        else nn.Conv2d(  # Normal convolution otherwise
                            out_channels * 2,  # From Skip connection
                            out_channels,
                            kernel_size=(3, 3),
                            padding=1,
                        )
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=(3, 3), padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(0.05),  # 5% dropout after each decoder block
                )
            )
            in_channels = out_channels
            dilation_rate //= 2  # Halve the dilation rate for the next block

        # Final convolution
        self.final_conv = nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        speech_beamformer_output: torch.Tensor,
        noise_beamformer_output: torch.Tensor,
        omni_input: torch.Tensor,
    ) -> torch.Tensor:
        # Convert complex inputs to magnitude if needed
        if speech_beamformer_output.is_complex():
            speech_beamformer_output = compute_magnitude(speech_beamformer_output)
        if noise_beamformer_output.is_complex():
            noise_beamformer_output = compute_magnitude(noise_beamformer_output)
        if omni_input.is_complex():
            omni_input = compute_magnitude(omni_input)

        x = torch.cat(
            (
                speech_beamformer_output.unsqueeze(1),
                noise_beamformer_output.unsqueeze(1),
                omni_input.unsqueeze(1),
            ),
            dim=1,
        )  # Concatenate along the channel dimension

        # Shape assertions
        self.check_specgram(x)

        # # Encoder forward pass
        skip_connections = []
        for i, (encoder, pool) in enumerate(
            zip(self.encoder_blocks, self.max_pool_layers)
        ):
            x = encoder(x)

            # Store skip connections before max pooling
            if i < len(self.encoder_blocks) - 1:
                skip_connections.append(x)

            # Apply max pooling
            x = pool(x)

        # # Decoder forward pass
        skip_connections = skip_connections[::-1]
        for i, decoder in enumerate(self.decoder_blocks):
            x = decoder[0](x)  # Transposed convolution

            x = torch.cat(
                (x, skip_connections[i]), dim=1
            )  # Concatenate with skip connection
            for layer in decoder[1:]:  # Apply remaining decoder layers
                x = layer(x)

        # Final convolution and sigmoid
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x.squeeze(1)  # Remove the channel dimension

    def check_specgram(self, x):
        """
        Asserts that the input shape is correct for the forward pass.

        Args:
            x: Input tensor. (Should be 4D with shape (batch_size, channels, height, width))

        Raises:
            ValueError: If the input shape is not correct.
        """
        if len(x.shape) != 4:
            raise ValueError(
                f"Expected input shape of 4 dimensions (batch_size, channels, height, width), got {len(x.shape)} dimensions."
            )
        if x.shape[1] != 3:
            raise ValueError(
                f"Expected input channels to be 3, got {x.shape[1]} channels."
            )


class Dilated_Unet_Full_Pipeline(nn.Module):
    """
    Wrapper class around the Dilated_Unet model, adding the beamforming and other components of the pipeline.
    """

    def __init__(
        self,
        dilated: bool = True,
        bf_style: str = "max_re",  # max_di, max_re, lc
        out_bf_style: str = "souden_mvdr",  # gevd_mwf, souden_mvdr
        no_out_beamformer: bool = False,
        train_on_masked_speech: bool = False,
        no_band_level_normalization: bool = True,
        compress_mag: bool = True,
    ):
        super().__init__()

        # Parse parameters
        self.dilated = dilated
        self.bf_style = bf_style
        self.out_bf_style = out_bf_style
        self.no_out_beamformer = no_out_beamformer
        self.train_on_masked_speech = train_on_masked_speech
        self.no_band_level_normalization = no_band_level_normalization
        self.compress_mag = compress_mag

        # Check parameters
        assert bf_style in [
            "max_di",
            "max_re",
            "lc",
        ], f"Invalid bf_style: {bf_style}. Must be one of 'max_di', 'max_re', 'lc'."
        assert out_bf_style in [
            "gevd_mwf",
            "souden_mvdr",
        ], f"Invalid out_bf_style: {out_bf_style}. Must be one of 'gevd_mwf', 'souden_mvdr'."

        # Unet
        self.dilated_unet = Dilated_Unet(dilated_mode=self.dilated)

        # Input beamformer
        self.input_beamformer = Ambisonic_Beamformer(
            bf_style=self.bf_style, sph_order=1, sph_implementation="cheind"
        )

        # Output beamformer
        if self.no_out_beamformer:
            self.output_beamformer = None
        else:
            self.output_beamformer = Ambisonic_Beamformer(
                bf_style=self.out_bf_style,
                sph_order=1,
                sph_implementation="cheind",
            )

    def forward(
        self,
        foa_mixture: torch.Tensor,
        speech_azimuth: torch.Tensor,
        speech_elevation: torch.Tensor,
        noise_azimuth: torch.Tensor,
        noise_elevation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the full pipeline.
        Args:
            foa_mixture: FOA mixture input. (batch_size, n_channels (4), f_bins, t_bins)
            speech_azimuth: Azimuth of the speech source. (batch_size,)
            speech_elevation: Elevation of the speech source. (batch_size,)
            noise_azimuth: Azimuth of the noise source. (batch_size,)
            noise_elevation: Elevation of the noise source. (batch_size,)
        Returns:
            estimated_spec: Estimated spectrogram. (batch_size, f_bins, t_bins)
        """
        # Beamform forward passes
        speech_beamformer_output = self.input_beamformer(
            foa_mixture,
            azimuth=speech_azimuth,
            elevation=speech_elevation,
            interference_azimuth=noise_azimuth,
            interference_elevation=noise_elevation,
        )  # (batch_size, f_bins, t_bins)
        noise_beamformer_output = self.input_beamformer(
            foa_mixture,
            azimuth=noise_azimuth,
            elevation=noise_elevation,
            interference_azimuth=speech_azimuth,
            interference_elevation=speech_elevation,
        )  # (batch_size, f_bins, t_bins)

        # Get magnitudes
        speech_beamformer_output_mag = compute_magnitude(speech_beamformer_output)
        noise_beamformer_output_mag = compute_magnitude(noise_beamformer_output)
        foa_mixture_w_mag = compute_magnitude(foa_mixture[:, 0, :, :])

        if self.compress_mag:
            # Compress magnitudes, inspired by deep filter net
            speech_beamformer_output_mag = speech_beamformer_output_mag**0.6
            noise_beamformer_output_mag = noise_beamformer_output_mag**0.6
            foa_mixture_w_mag = foa_mixture_w_mag**0.6

        # Sequence-dependent, band-level normalization of each
        # feature (improves WER according to paper)
        if self.no_band_level_normalization:
            pass
        else:
            speech_beamformer_output_mag = band_level_normalize(
                speech_beamformer_output_mag
            )
            noise_beamformer_output_mag = band_level_normalize(
                noise_beamformer_output_mag
            )

        # Mask estimator forward pass
        speech_mask = self.dilated_unet(
            speech_beamformer_output_mag,
            noise_beamformer_output_mag,
            foa_mixture_w_mag,
        )  # (batch_size, f_bins, t_bins)

        if self.compress_mag:
            # Decompress magnitudes
            speech_mask = speech_mask ** (1 / 0.6)

        # Mask management
        noise_mask = torch.ones_like(speech_mask) - speech_mask
        speech_mask = speech_mask.type(torch.complex64)
        noise_mask = noise_mask.type(torch.complex64)

        # Apply masks to FOA input (uniform application along channels)
        masked_speech = speech_mask.unsqueeze(1) * foa_mixture
        masked_noise = noise_mask.unsqueeze(1) * foa_mixture

        # Out beamformer forward pass
        # No need to apply out beamformer if we're training on masked speech
        # or if we're not using it
        if self.no_out_beamformer or (self.train_on_masked_speech and self.training):
            estimated_spec = masked_speech[:, 0, :, :]
        else:
            output_bf_spec = self.output_beamformer(
                foa_mixture,  # TODO try using masked speech here !!
                azimuth=speech_azimuth,
                elevation=speech_elevation,
                interference_azimuth=noise_azimuth,
                interference_elevation=noise_elevation,
                speech_specgram=masked_speech,
                noise_specgram=masked_noise,
            )  # (batch_size, f_bins, t_bins)
            estimated_spec = output_bf_spec

        return estimated_spec  # (batch_size, f_bins, t_bins)


# ----------------------------------------------------------
# Factory functions to create instances of the
# Dilated_Unet_Full_Pipeline with different configurations
# ----------------------------------------------------------


def get_dilated_unet_full_pipeline_default():
    """With default config from the paper."""
    return Dilated_Unet_Full_Pipeline(
        dilated=True,
        bf_style="lc",
        out_bf_style="gevd_mwf",
        no_out_beamformer=False,
        train_on_masked_speech=True,
        no_band_level_normalization=False,
        compress_mag=False,
    )


def get_dilated_unet_full_pipeline_best():
    """With best config from my experiments."""
    return Dilated_Unet_Full_Pipeline(
        dilated=True,
        bf_style="max_re",
        out_bf_style="souden_mvdr",
        no_out_beamformer=False,
        train_on_masked_speech=False,
        no_band_level_normalization=True,
        compress_mag=True,
    )


def get_dilated_unet_full_pipeline_from_config(config: dict):
    return Dilated_Unet_Full_Pipeline(**config)
