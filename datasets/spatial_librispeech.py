"""
This script provides functionality to download, process, and manage the Spatial Librispeech dataset, merged with the LibriSpeech dataset.
This might need some cleaning up, but it works for now.
Example usage is provdided in the main function at the bottom of the script.
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal
from urllib.request import urlretrieve

import librosa
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm

from tools.gcc_phat import gcc_phat
from tools.stft_utils import stft

# ----------------------------------------------------------
# Download Spatial Librispeech dataset + Librispeech
# ----------------------------------------------------------


# Dataset download link
SLS_URI = "https://docs-assets.developer.apple.com/ml-research/datasets/spatial-librispeech/v1"


def _download_SLS_sample(sample_id: int, local_path="./datasets/SpatialLibrispeech"):
    """Downloads one sample from the Spatial LibriSpeech dataset onto the local machine."""
    sample_name = f"{sample_id:06}.flac"
    local_path = Path(local_path)
    download_targets = {
        (local_path / "ambisonics" / sample_name): (
            SLS_URI + "/ambisonics/" + sample_name
        ),
        (local_path / "noise_ambisonics" / sample_name): (
            SLS_URI + "/noise_ambisonics/" + sample_name
        ),
    }

    for local_filename, url in download_targets.items():
        if not local_filename.exists():
            local_filename.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(url, filename=local_filename)
            print(f"Downloaded {local_filename}")
        else:
            print(f"File already exists: {local_filename}, skipping download.")


def _download_multiple_SLS_samples(
    sample_ids, local_path="./datasets/SpatialLibrispeech", max_workers=5
):
    """Downloads multiple samples concurrently using threading."""
    local_path = Path(local_path)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_SLS_sample, sample_id, local_path): sample_id
            for sample_id in sample_ids
        }

        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error downloading sample {sample_id}: {e}")
    print("Multiple samples downloaded.")


def download_lite_SLS_dataset(
    sls_data_dir="./datasets/SpatialLibrispeech/",
    train_head: int | None = 100,
    test_head: int | None = 10,
):
    """Downloads heads of lite SLS dataset."""

    columns = ["sample_id", "split", "lite_version"]
    sls_data_dir = Path(sls_data_dir)

    # Download metadata
    metadata_path = sls_data_dir / "metadata.parquet"
    if not metadata_path.exists():
        print(f"Downloading metadata.parquet")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(SLS_URI + "/metadata.parquet", filename=metadata_path)
        print(f"Downloaded metadata.parquet")
    else:
        print(f"Metadata already downloaded, skipping download.")

    # Load metadata
    df_metadata = pd.read_parquet(
        sls_data_dir.joinpath("metadata.parquet"), columns=columns
    )

    # filter df to only include lite_version == True
    df_metadata = df_metadata[df_metadata["lite_version"] == True]
    df_metadata_train = df_metadata[df_metadata["split"] == "train"]
    df_metadata_test = df_metadata[df_metadata["split"] == "test"]

    if train_head is not None:
        df_metadata_train = df_metadata_train.head(train_head)
    if test_head is not None:
        df_metadata_test = df_metadata_test.head(test_head)

    del df_metadata

    samples_to_download = (
        df_metadata_train["sample_id"].to_list()
        + df_metadata_test["sample_id"].to_list()
    )

    _download_multiple_SLS_samples(samples_to_download, local_path=sls_data_dir)


def _convert_flac_to_wav(flac_path: Path):
    """Converts a single FLAC file to WAV."""
    wav_path = flac_path.with_suffix(".wav")
    if not wav_path.exists():
        try:
            subprocess.run(["ffmpeg", "-i", str(flac_path), str(wav_path)], check=True)
            print(f"Converted {flac_path} to {wav_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {flac_path}: {e}")
    else:
        print(f"File already converted: {flac_path}")


def convert_all_flac_in_directory(sls_data_dir: str = "./datasets/SpatialLibrispeech/"):
    """Converts all FLAC files in the sls_data_dir to WAV files."""
    flac_files = Path(sls_data_dir).rglob(
        "*.flac"
    )  # Find all FLAC files in the directory
    with ThreadPoolExecutor() as executor:
        executor.map(_convert_flac_to_wav, flac_files)
    print("All flac converted to wav.")


def _get_ls_dataset_metadata_df(dataset: LIBRISPEECH, name):
    metadata_list = []
    for i in tqdm(range(len(dataset)), desc=f"Processing {name}", unit="file"):
        # k += 1
        audio_path, sample_rate, _, speaker_id, chapter_id, utterance_id = (
            dataset.get_metadata(i)
        )
        metadata = {
            "audio_path": Path(audio_path).as_posix(),  # Convert Path object to string
            "reader_id": speaker_id,
            "chapter_id": chapter_id,
            "sequence_number": utterance_id,
        }
        metadata["subset"] = name
        metadata_list.append(metadata)
    return pd.DataFrame(metadata_list)


def _get_local_sls_dataset_metadata_df(sls_data_dir: Path):
    columns = [
        "sample_id",
        "split",
        "lite_version",
        "audio_info/duration",
        "audio_info/frames",
        "audio_info/size/ambisonics",
        "audio_info/size/noise_ambisonics",
        "speech/azimuth",
        "speech/elevation",
        "speech/distance",
        "noise/azimuth",
        "noise/elevation",
        "noise/distance",
        "speech/librispeech_metadata/chapter_id",
        "speech/librispeech_metadata/reader_id",
        "speech/librispeech_metadata/sequence_number",
        "speech/librispeech_metadata/subset",
    ]

    # Load metadata
    sls_metadata_df = pd.read_parquet(
        sls_data_dir / "metadata.parquet",
        columns=columns,
    )

    # Find all locally downloaded data
    ambisonic_ids = {
        int(path.stem) for path in (sls_data_dir / "ambisonics").glob("*.wav")
    }
    noise_ambisonic_ids = {
        int(path.stem) for path in (sls_data_dir / "noise_ambisonics").glob("*.wav")
    }
    downloaded_ids = ambisonic_ids & noise_ambisonic_ids

    # Filter non-present rows
    sls_metadata_df = sls_metadata_df[sls_metadata_df["sample_id"].isin(downloaded_ids)]

    print("Loaded metadata pertinent to downloaded SLS files")

    return sls_metadata_df


def download_LS_dataset_and_merge_metadata_with_SLS(
    sls_data_dir=Path("datasets") / "SpatialLibrispeech",
    ls_data_dir=Path("datasets") / "LibriSpeech",
    reset_local_sls_metadata: bool = False,
):
    """
    Downloads the LibriSpeech dataset and merges its metadata with the Spatial Librispeech dataset.
    This function will download the LibriSpeech dataset subsets, generate metadata for them,
    and then merge this metadata with the Spatial Librispeech dataset metadata.
    The merged metadata will be saved to a parquet file.
    """
    # Convert input paths to Path objects
    sls_data_dir = Path(sls_data_dir)
    ls_data_dir = Path(ls_data_dir)

    # Download / load the LibriSpeech dataset subsets
    ls_data_dir.mkdir(parents=True, exist_ok=True)
    subset_names = ["train-clean-100", "train-clean-360", "test-clean", "test-other"]
    print("Looking for LibriSpeech dataset in", ls_data_dir)
    ls_datasets = {
        name: LIBRISPEECH(ls_data_dir, url=name, download=True) for name in subset_names
    }
    print("LibriSpeech dataset found.")

    # ---- Build / Get Librispeech metadata ----
    ls_metadata_path = ls_data_dir / "ls_metadata.csv"
    if ls_metadata_path.exists():
        ls_metadata_df = pd.read_csv(ls_metadata_path)
    else:
        print("Librispeech metadata file does not exist. Generating new metadata file.")

        # Get metadata from each dataset
        dfs = {}
        for subset_name in subset_names:
            dfs[subset_name] = _get_ls_dataset_metadata_df(
                ls_datasets[subset_name], subset_name
            )

        # Combine all metadata into one large dataframe
        ls_metadata_df = pd.concat(list(dfs.values()), ignore_index=True)

        # Save the librispeech metadata DataFrame to a CSV file
        ls_metadata_df.to_csv(ls_metadata_path, index=False, header=True)

    # ---- Load local Spatial Librispeech metadata ----
    local_sls_metadata_path = sls_data_dir / "local_sls_metadata.csv"
    if local_sls_metadata_path.exists() and not reset_local_sls_metadata:
        sls_metadata_df = pd.read_csv(local_sls_metadata_path)
    else:
        print(
            "Local Spatial Librispeech metadata file does not exist. Generating new metadata file."
        )

        sls_metadata_df = _get_local_sls_dataset_metadata_df(sls_data_dir)
        sls_metadata_df.to_csv(local_sls_metadata_path, index=False, header=True)

    # ---- Combine Librispeech metadata with Spatial Librispeech metadata ----
    local_sls_with_ls_metadata_path = (
        sls_data_dir / "local_sls_with_ls_metadata.parquet"
    )
    if local_sls_with_ls_metadata_path.exists() and not reset_local_sls_metadata:
        print("Local SLS with LS metadata file already exists. Skipping merge.")
        return
    else:
        print("Merging Librispeech metadata with local SLS metadata.")

        # Add a clean_audio_path column to the spatial librispeech metadata
        # by merging the librispeech metadata with the spatial librispeech metadata
        # on these columns:
        # - "reader_id" on "speech/librispeech_metadata/reader_id"
        # - "chapter_id" on "speech/librispeech_metadata/chapter_id"
        # - "sequence_number" on "speech/librispeech_metadata/sequence_number"
        # - "subset" on "speech/librispeech_metadata/subset"
        #
        # and add "audio_path" into the new column "speech/librispeech_metadata/audio_path"

        # Rename columns in ls_metadata_df to match merge keys
        ls_metadata_df_renamed = ls_metadata_df.rename(
            columns={
                "reader_id": "speech/librispeech_metadata/reader_id",
                "chapter_id": "speech/librispeech_metadata/chapter_id",
                "sequence_number": "speech/librispeech_metadata/sequence_number",
                "subset": "speech/librispeech_metadata/subset",
                "audio_path": "speech/librispeech_metadata/audio_path",
            }
        )

        # Perform the merge
        merged_sls_df = sls_metadata_df.merge(
            ls_metadata_df_renamed[
                [
                    "speech/librispeech_metadata/reader_id",
                    "speech/librispeech_metadata/chapter_id",
                    "speech/librispeech_metadata/sequence_number",
                    "speech/librispeech_metadata/subset",
                    "speech/librispeech_metadata/audio_path",
                ]
            ],
            on=[
                "speech/librispeech_metadata/reader_id",
                "speech/librispeech_metadata/chapter_id",
                "speech/librispeech_metadata/sequence_number",
                "speech/librispeech_metadata/subset",
            ],
            how="left",
        )

        # Check for any missing matches
        missing_audio_paths = (
            merged_sls_df["speech/librispeech_metadata/audio_path"].isna().sum()
        )
        print(f"Number of records without matched audio paths: {missing_audio_paths}")
        print(
            f"Number of records with matched audio paths: {len(merged_sls_df) - missing_audio_paths}"
        )

        # Save the merged metadata DataFrame to a parquet file
        output_path = sls_data_dir / "local_sls_with_ls_metadata.parquet"
        merged_sls_df.to_parquet(output_path, index=False)
        print(f"Saved merged metadata to {output_path}")


def add_gcc_phat_to_merged_metadata(
    sls_data_dir: str = "./datasets/SpatialLibrispeech/",
    ls_data_dir: str = "./datasets/LibriSpeech/",
):
    """
    Computes the gcc phat for each sample in the merged metadata and adds it to the metadata.
    """
    # Convert input paths to Path objects
    sls_data_dir = Path(sls_data_dir)
    ls_data_dir = Path(ls_data_dir)
    with_gcc_phat_path = (
        sls_data_dir / "local_sls_with_ls_metadata_with_gcc_phat.parquet"
    )

    if with_gcc_phat_path.exists():
        print(
            f"Metadata with gcc phat already exists at {with_gcc_phat_path}. Skipping computation."
        )
        return
    else:
        # Load the merged metadata
        df_metadata = pd.read_parquet(
            sls_data_dir / "local_sls_with_ls_metadata.parquet"
        )

        # Initialize a list for the gcc phat
        gcc_phat_list = {"uid": [], "gcc_phat": []}

        # Function to process a single item
        def process_item(row):
            uid = row["sample_id"]
            clean_audio_path = row["speech/librispeech_metadata/audio_path"]

            # Load the full clean audio and the full ambisonic audio
            audio_arrays = SpatialLibrispeechDataset.simple_get_audios(
                uid,
                clean_audio_path,
                sample_rate=16000,
                sls_data_dir=sls_data_dir,
                ls_data_dir=ls_data_dir,
                duration=5,  # seconds
            )

            # compute the gcc phat
            tau, _ = gcc_phat(
                audio_arrays["ambisonic"], audio_arrays["clean_audio"], fs=16000
            )
            tau = -tau
            tau_frames = int(tau * 16000)

            return uid, tau_frames

        # Process items in parallel
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_row = {
                executor.submit(process_item, row): i
                for i, row in df_metadata.iterrows()
            }

            for future in tqdm(
                as_completed(future_to_row),
                total=len(df_metadata),
                desc="Computing GCC PHAT for local SLS dataset",
            ):
                try:
                    uid, tau_frames = future.result()
                    results.append((uid, tau_frames))
                except Exception as exc:
                    print(f"Item generated an exception: {exc}")

        # Convert results to dictionary
        gcc_phat_dict = {uid: tau_frames for uid, tau_frames in results}

        # Map results back to dataframe using sample_id
        print("Mapping results back to dataframe")
        df_metadata["gcc_phat"] = df_metadata["sample_id"].map(gcc_phat_dict)

        # Save the dataframe to a parquet file
        df_metadata.to_parquet(with_gcc_phat_path, index=False)
        print(f"Saved metadata with gcc phat to {with_gcc_phat_path}")


# ----------------------------------------------------------
# Dataset + Datamodule
# ----------------------------------------------------------


class SpatialLibrispeechDataset(Dataset):
    def __init__(
        self,
        sls_data_dir: str = "./datasets/SpatialLibrispeech/",
        ls_data_dir: str = "./datasets/LibriSpeech/",
        stage: Literal["train", "test"] | None = "train",
        random_audio_offset: bool = True,
        tiny: bool = False,
    ):
        """
        Spatial Librispeech Dataset.
        Args:
            sls_data_dir (str, optional): Path to the Spatial Librispeech dataset directory.
                Defaults to "./datasets/SpatialLibrispeech/".
            ls_data_dir (str, optional): Path to the LibriSpeech dataset directory.
                Defaults to "./datasets/LibriSpeech/".
            stage (Literal["train", "test"] | None, optional): The stage of the dataset to load
                (train or test). Defaults to "train". If None, all data is loaded.
            random_audio_offset (bool, optional): Whether to randomly offset audio samples.
                This provides more diversity while training, but also is inconsistent.
                Defaults to True.
            tiny (bool, optional): Whether to use a tiny subset of the dataset (first 100 samples).
                Defaults to False.
        """
        super().__init__()

        print("Initializing Spatial Librispeech Dataset.")

        # General info -----------------------------------------------------

        self.sample_rate = 16000
        self.sound_speed = 343.0  # Speed of sound in m/s
        self.num_ambisonic_channels = 4  # Ambisonic channels
        self.num_noise_ambisonic_channels = 4  # Ambisonic channels
        self.sample_length = 1.25  # seconds
        self.min_mean_audio_energy = 1e-4  # Minimum mean audio energy
        self.sls_data_dir = Path(sls_data_dir)
        self.ls_data_dir = Path(ls_data_dir)

        self.random_audio_offset = random_audio_offset
        self.print_audio_load_errors = False

        # Load Spatial Librispeech -----------------------------------------

        # Load metadata
        columns = [
            "sample_id",
            "split",
            # "lite_version",
            "audio_info/duration",
            "audio_info/frames",
            # "audio_info/size/ambisonics",
            # "audio_info/size/noise_ambisonics",
            "speech/azimuth",
            "speech/elevation",
            "speech/distance",
            "noise/azimuth",
            "noise/elevation",
            "noise/distance",
            # "speech/librispeech_metadata/chapter_id",
            # "speech/librispeech_metadata/reader_id",
            # "speech/librispeech_metadata/sequence_number",
            # "speech/librispeech_metadata/subset",
            "speech/librispeech_metadata/audio_path",
            "gcc_phat",
        ]

        self.df_metadata = pd.read_parquet(
            self.sls_data_dir / "local_sls_with_ls_metadata_with_gcc_phat.parquet",
            columns=columns,
        )
        print("Loaded metadata")

        # Select split
        if stage is not None:
            self.df_metadata = self.df_metadata[self.df_metadata["split"] == stage]
            self.df_metadata = self.df_metadata.drop(columns=["split"])

        # Filter samples that are too short
        self.df_metadata = self.df_metadata[
            self.df_metadata["audio_info/duration"] >= self.sample_length
        ]

        # Eliminate all rows with NaN values in the gcc phat column
        self.df_metadata = self.df_metadata[~self.df_metadata["gcc_phat"].isna()]

        # If tiny is True, take a subset of the dataset
        if tiny:
            self.df_metadata = self.df_metadata.head(100)
            print("Using tiny version of the dataset.")

        # Convert column types once
        self.df_metadata = self.df_metadata.astype(
            {
                "sample_id": int,
                "speech/azimuth": float,
                "speech/elevation": float,
                "speech/distance": float,
                "noise/azimuth": float,
                "noise/elevation": float,
                "noise/distance": float,
                "audio_info/duration": float,
                "audio_info/frames": int,
                "speech/librispeech_metadata/audio_path": str,
                "gcc_phat": int,
            }
        )

        print("Filtered metadata")

    def __len__(self):
        return len(self.df_metadata)

    def __str__(self):
        """Returns a string representation of the dataset information."""
        info = [
            "Spatial Librispeech Dataset information:",
            f"Data directory: {self.sls_data_dir}",
            f"Number of samples: {len(self.df_metadata)}",
            f"Sample length: {self.sample_length} seconds",
            f"Sample rate: {self.sample_rate}",
            f"Number of ambisonic channels: {self.num_ambisonic_channels}",
            f"Number of noise ambisonic channels: {self.num_noise_ambisonic_channels}",
        ]
        return "\n".join(info)

    def _get_audios(self, uid, clean_audio_path, gcc_phat_delay):
        """
        Returns a random slice of the audio data for a given sample ID.
        Loops the audio if the length is not correct.
        This is a workaround for the issue of audio length mismatch.

        args:
            uid: sample ID
            clean_audio_path: path to the clean audio file
            gcc_phat_delay: delay in frames to apply to the ambisonic audio
                (converted to seconds later)
        returns:
            ambisonic_audio: ambisonic audio tensor
            noise_ambisonic_audio: noise ambisonic audio tensor
            clean_audio: clean audio tensor
        """
        # Dictionary to store audio paths and tensors
        audio_paths = {
            "ambisonic": str(self.sls_data_dir / "ambisonics" / f"{uid:06}.wav"),
            "noise_ambisonic": str(
                self.sls_data_dir / "noise_ambisonics" / f"{uid:06}.wav"
            ),
            "clean_audio": str(self.ls_data_dir / "LibriSpeech" / clean_audio_path),
        }

        # Some inits
        audio_tensors = {
            "ambisonic": None,
            "noise_ambisonic": None,
            "clean_audio": None,
        }
        desired_length_frames = int(self.sample_length * self.sample_rate)

        # Get starts
        audio_actual_start = {
            "ambisonic": None,
            "noise_ambisonic": 0,
            "clean_audio": None,
        }
        if gcc_phat_delay < 0:
            audio_actual_start["ambisonic"] = abs(gcc_phat_delay) / self.sample_rate
            audio_actual_start["clean_audio"] = 0.0
        else:
            audio_actual_start["ambisonic"] = 0.0
            audio_actual_start["clean_audio"] = abs(gcc_phat_delay) / self.sample_rate

        # Get durations
        audio_duration = {}
        for audio_name, file_path in audio_paths.items():
            audio_duration[audio_name] = librosa.get_duration(
                path=file_path, sr=self.sample_rate
            )

        # Get actual length of the speech audios
        min_duration = max(
            min(
                audio_duration["ambisonic"] - audio_actual_start["ambisonic"],
                audio_duration["clean_audio"] - audio_actual_start["clean_audio"],
            ),
            0,
        )

        # Get random offsets to select random slices of length sample_length
        rand_offset = {"ambisonic": None, "noise_ambisonic": None, "clean_audio": None}

        if self.random_audio_offset:
            rand_offset_speech = max(
                torch.rand(1).item() * (min_duration - self.sample_length), 0.0
            )
            rand_offset_noise = max(
                torch.rand(1).item()
                * (audio_duration["noise_ambisonic"] - self.sample_length),
                0.0,
            )
        else:
            rand_offset_speech = 0.0
            rand_offset_noise = 0.0

        rand_offset["ambisonic"] = rand_offset_speech + audio_actual_start["ambisonic"]
        rand_offset["clean_audio"] = (
            rand_offset_speech + audio_actual_start["clean_audio"]
        )
        rand_offset["noise_ambisonic"] = (
            audio_actual_start["noise_ambisonic"] + rand_offset_noise
        )

        for audio_name, file_path in audio_paths.items():
            # Load with offset and duration
            audio_data = librosa.load(
                file_path,
                sr=self.sample_rate,
                mono=False,
                offset=rand_offset[audio_name],
                duration=self.sample_length,
            )

            # Librosa returns a tuple (audio_data, sample_rate)
            audio_data = audio_data[0]

            if audio_name == "clean_audio" or audio_data.ndim == 1:
                # If the audio is mono, add a channel dimension
                # Little hack to make sure the looping code works for mono audio (aka clean audio)
                audio_data = np.expand_dims(audio_data, axis=0)

            # If loaded audio is not of correct length, loop until big enough
            if audio_data.shape[1] != desired_length_frames:
                if audio_name == "clean_audio" or audio_name == "ambisonic":
                    # Little hack to make sure that both audio files are the same length before looping
                    audio_data = audio_data[:, : int(audio_duration * self.sample_rate)]
                print(
                    f"HEY! Audio length mismatch for {audio_name} sample {uid}. Looping audio."
                )
                while audio_data.shape[1] < desired_length_frames:
                    audio_data = np.concatenate(
                        (
                            audio_data,
                            audio_data[
                                :, : desired_length_frames - audio_data.shape[1]
                            ],
                        ),
                        axis=1,
                    )

            # Convert to tensor
            audio_tensors[audio_name] = torch.tensor(audio_data).float()

        # Extract the tensors
        ambisonic_audio = audio_tensors["ambisonic"]
        noise_ambisonic_audio = audio_tensors["noise_ambisonic"]
        clean_audio = audio_tensors["clean_audio"].squeeze(0)

        return ambisonic_audio, noise_ambisonic_audio, clean_audio

    def _test_audios(self, ambisonic_audio, noise_ambisonic_audio, clean_audio, uid):
        # Test if audio is the right length
        expected_length = int(self.sample_length * self.sample_rate)
        if (
            ambisonic_audio.shape[1] != expected_length
            or noise_ambisonic_audio.shape[1] != expected_length
            or len(clean_audio) != expected_length
        ):
            raise ValueError(
                f"Audio length mismatch for sample {uid}: speech {ambisonic_audio.shape[1]} "
                f"!= noise {noise_ambisonic_audio.shape[1]} "
                f"!= clean {len(clean_audio)} "
                f"!= target {expected_length}"
            )
        # Test if audio contains NaN values
        if (
            torch.isnan(ambisonic_audio).any()
            or torch.isnan(noise_ambisonic_audio).any()
            or torch.isnan(clean_audio).any()
        ):
            raise ValueError(
                f"Audio contains NaN values for sample {uid}: "
                f"{torch.isnan(ambisonic_audio).any()} "
                f"or {torch.isnan(noise_ambisonic_audio).any()} "
                f"or {torch.isnan(clean_audio).any()}"
            )

        # Test if clean audio contains nearly no energy.
        # Don't do this if not selecting audio with random offsets, to avoid infinite test loops.
        if self.random_audio_offset:
            mean_amb_energy = torch.mean(clean_audio**2)
            if mean_amb_energy < self.min_mean_audio_energy:
                raise ValueError(
                    f"Clean audio contains nearly no energy for sample {uid}: {mean_amb_energy}"
                )

    def __getitem__(self, idx):
        """
        Returns a single SLS datapoint.
        This includes signal audio, noise audio, and clean audio along with metadata.
        """
        # Get metadata
        row_metadata = self.df_metadata.iloc[idx]
        row_metadata = row_metadata.to_dict()

        uid = row_metadata["sample_id"]
        clean_audio_path = Path(row_metadata["speech/librispeech_metadata/audio_path"])
        gcc_phat_delay = row_metadata["gcc_phat"]

        # Try getting audio
        tries = 0
        max_tries = 100
        last_audioload_exception = ""
        while True:
            try:
                # Handle reloading audios in case tests fail
                tries += 1
                if tries > max_tries:
                    print(
                        "Max dataset load tries exceeded.\n"
                        "This is probably due to the minimum mean audio energy threshold being too high.\n"
                        "Lowering the threshold."
                    )
                    self.min_mean_audio_energy = self.min_mean_audio_energy * 0.9
                    if tries > max_tries * 10:
                        print(
                            "Max dataset load tries REALLY exceeded. There must be another bug.\n"
                            f"Last audio load exception: {last_audioload_exception}\n"
                            "Exiting."
                        )
                        exit()

                # Get audios
                ambisonic_audio, noise_ambisonic_audio, clean_audio = self._get_audios(
                    uid, clean_audio_path, gcc_phat_delay
                )  # , clean_audio_frame_delay)
                self._test_audios(
                    ambisonic_audio, noise_ambisonic_audio, clean_audio, uid
                )
                break  # Exit loop if successful
            except Exception as e:
                last_audioload_exception = e
                if self.print_audio_load_errors:
                    print(f"Error loading audio for sample {uid}: {e}. Trying again...")
                pass

        return {
            "foa_speech": ambisonic_audio,
            "foa_noise": noise_ambisonic_audio,
            "clean_audio": clean_audio,
            "metadata": row_metadata,
        }

    @staticmethod
    def simple_get_audios(
        uid: int,
        clean_audio_path: str,
        sample_rate: int = 16000,
        sls_data_dir: str = "./datasets/SpatialLibrispeech/",
        ls_data_dir: str = "./datasets/LibriSpeech/",
        duration: float = 1.25,  # seconds
    ):
        """
        Static method used in add_gcc_phat_to_merged_metadata to load the audio files for gcc phat computation.
        """
        # Dictionary to store audio paths and tensors
        audio_paths = {
            "ambisonic": str(sls_data_dir / "ambisonics" / f"{uid:06}.wav"),
            # "noise_ambisonic": str(
            #     sls_data_dir / "noise_ambisonics" / f"{uid:06}.wav"
            # ),
            "clean_audio": str(ls_data_dir / "LibriSpeech" / clean_audio_path),
        }

        audio_arrays = {}

        # Load with offset and duration
        for audio_name, file_path in audio_paths.items():
            audio_data, _ = librosa.load(
                file_path,
                sr=sample_rate,
                mono=False,
                duration=duration,
            )
            if audio_data.ndim == 2:
                # If the audio is ambisonic, take only w channel
                audio_data = audio_data[0, :]
            audio_arrays[audio_name] = audio_data

        return audio_arrays

    @staticmethod
    def parse_datapoint(datapoint, device="cpu", mixture_snr_db=20, return_stft=True):
        """
        Parses a single or batch of datapoints from the Spatial Librispeech dataset.
        This includes the following steps:
        1. Convert the inputs to tensors and move them to the specified device.
        2. Normalize the energy values of speech and noise to match the target mixture SNR.
        3. Create the mixture by summing the speech and noise signals.
        4. Optionally, convert the signals to the STFT domain.
        5. Return the processed signals and metadata.

        Parameters:
        - datapoint: A single or batch of datapoints from the dataset.
        - device: The device to move the tensors to (default: "cpu").
        - mixture_snr_db: The target SNR in dB for the mixture (default: 20 dB, as in guerin paper).
        - return_stft: Whether to return the STFT of the signals (default: True).

        """
        # Foa
        foa_speech = datapoint["foa_speech"]
        foa_noise = datapoint["foa_noise"]
        clean_speech = datapoint["clean_audio"]

        assert not torch.isnan(foa_speech).any(), "NaNs in input foa_speech"
        assert not torch.isnan(foa_noise).any(), "NaNs in input foa_noise"

        # Metadata
        metadata = datapoint["metadata"]
        speech_azimuth = (
            torch.tensor([metadata["speech/azimuth"]])
            if isinstance(metadata["speech/azimuth"], float)
            else metadata["speech/azimuth"]
        )
        speech_elevation = (
            torch.tensor([metadata["speech/elevation"]])
            if isinstance(metadata["speech/elevation"], float)
            else metadata["speech/elevation"]
        )
        noise_azimuth = (
            torch.tensor([metadata["noise/azimuth"]])
            if isinstance(metadata["noise/azimuth"], float)
            else metadata["noise/azimuth"]
        )
        noise_elevation = (
            torch.tensor([metadata["noise/elevation"]])
            if isinstance(metadata["noise/elevation"], float)
            else metadata["noise/elevation"]
        )

        # ---- Move data to device ------------------------------------------------

        foa_speech = foa_speech.to(device)
        foa_noise = foa_noise.to(device)
        clean_speech = clean_speech.to(device)
        speech_azimuth = speech_azimuth.to(device)
        speech_elevation = speech_elevation.to(device)
        noise_azimuth = noise_azimuth.to(device)
        noise_elevation = noise_elevation.to(device)

        # ---- Prepare noise and mixture / SNR ------------------------------------

        # Calculate power using only the W channel (0) but ensure dimensions align
        mean_speech_power = torch.mean(foa_speech[..., 0, :] ** 2, dim=-1)
        mean_speech_power = torch.clamp(
            mean_speech_power, min=1e-8
        )  # Avoid division by zero
        mean_speech_power = mean_speech_power.unsqueeze(-1).unsqueeze(-1)

        foa_speech = foa_speech / torch.sqrt(mean_speech_power)

        mean_noise_power = torch.mean(foa_noise[..., 0, :] ** 2, dim=-1)
        mean_noise_power = torch.clamp(
            mean_noise_power, min=1e-8
        )  # Avoid division by zero
        mean_noise_power = mean_noise_power.unsqueeze(-1).unsqueeze(-1)

        foa_noise = foa_noise / torch.sqrt(mean_noise_power)

        target_noise_power = mean_speech_power / (10 ** (mixture_snr_db / 10))
        foa_noise = foa_noise * torch.sqrt(target_noise_power)

        # Sum speech and noise to create the mixture
        foa_mixture = foa_speech + foa_noise

        if return_stft:
            foa_speech = stft(foa_speech)
            foa_noise = stft(foa_noise)
            foa_mixture = stft(foa_mixture)
            clean_speech = stft(clean_speech)

        return (
            foa_speech,
            foa_noise,
            foa_mixture,
            clean_speech,
            speech_azimuth,
            speech_elevation,
            noise_azimuth,
            noise_elevation,
        )


class SpatialLibrispeechDataModule(LightningDataModule):
    def __init__(
        self,
        sls_data_dir: str = "./datasets/SpatialLibrispeech/",
        ls_data_dir: str = "./datasets/LibriSpeech/",
        batch_size=1,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        tiny: bool = False,
        random_audio_offset: bool = True,
    ):
        super().__init__()
        self.sls_data_dir = Path(sls_data_dir)
        self.ls_data_dir = Path(ls_data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.tiny = tiny
        self.random_audio_offset = random_audio_offset
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(
        self,
        train_head: int | None = 100,
        test_head: int | None = 10,
        reset_local_sls_metadata=False,
    ):
        download_lite_SLS_dataset(self.sls_data_dir, train_head, test_head)
        convert_all_flac_in_directory(self.sls_data_dir)
        download_LS_dataset_and_merge_metadata_with_SLS(
            self.sls_data_dir,
            self.ls_data_dir,
            reset_local_sls_metadata=reset_local_sls_metadata,
        )
        add_gcc_phat_to_merged_metadata(self.sls_data_dir, self.ls_data_dir)

    def setup(self, stage: str = None):
        # called on every process in DDP
        if stage == "train" or stage == None:
            train_dataset = SpatialLibrispeechDataset(
                stage="train",
                tiny=self.tiny,
                random_audio_offset=self.random_audio_offset,
            )
            self.train, self.val = data.random_split(
                train_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
            )
        if stage == "test" or stage == None:
            self.test = SpatialLibrispeechDataset(
                stage="test",
                tiny=self.tiny,
                random_audio_offset=self.random_audio_offset,
            )

    def train_dataloader(self):
        if self.train is None:
            self.setup("train")
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        if self.val is None:
            self.setup("train")
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test is None:
            self.setup("test")
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def on_exception(self, exception):
        # clean up state after the trainer faced an exception
        print(exception)

    def teardown(self):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        pass


# ----------------------------------------------------------
# Example usage
# ----------------------------------------------------------


if __name__ == "__main__":
    # Please make sure to set the paths to your own directories.
    datamodule = SpatialLibrispeechDataModule(batch_size=2)

    # ---- Preparing the dataset ----

    print("\nPrepare the dataset\n")

    # Use None to load the entire lite dataset
    datamodule.prepare_data(
        train_head=100, test_head=100, reset_local_sls_metadata=True
    )

    # ---- Load the dataset and test it ----

    print("\nDesting the dataset\n")

    dataloader = datamodule.train_dataloader()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    import matplotlib.pyplot as plt

    for batch in dataloader:
        # parse datapoint
        (
            foa_speech,  # (batch_size, n_channels, f_bins, t_bins)
            foa_noise,  # (batch_size, n_channels, f_bins, t_bins)
            foa_mixture,  # (batch_size, n_channels, f_bins, t_bins)
            clean_speech,  # (batch_size, f_bins, t_bins)
            speech_azimuth,  # (batch_size)
            speech_elevation,  # (batch_size)
            noise_azimuth,  # (batch_size)
            noise_elevation,  # (batch_size)
        ) = SpatialLibrispeechDataset.parse_datapoint(
            batch, device=DEVICE, mixture_snr_db=-15.0, return_stft=False
        )

        # Convert to numpy for plotting
        foa_speech = foa_speech[-1, 0, :].cpu().numpy()
        foa_noise = foa_noise[-1, 0, :].cpu().numpy()
        clean_speech = clean_speech[-1, :].cpu().numpy()

        tau, cc = gcc_phat(foa_speech, clean_speech, fs=16000)
        tau = -tau
        tau_frames = int(tau * 16000)
        print(f"Estimated time delay: {tau:.6f} seconds")
        print(f"Estimated time delay (samples): {tau_frames} samples")
        print("cross correlation :", cc)

        plt.plot(foa_speech, label="FOA Speech (W channel)", alpha=0.5)
        plt.plot(foa_noise, label="FOA Noise (W channel)", alpha=0.5)
        plt.plot(clean_speech, label="Clean Speech", alpha=0.5)
        plt.plot(clean_speech[tau_frames:], label="Clean Speech, shifted", alpha=0.5)
        plt.legend()
        plt.title("FOA Speech, FOA Noise and Clean Speech")
        plt.show()
