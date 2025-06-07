# ambispeechkit

PRELIMINARY WARNING: THIS IS A VERY UNDER DEVELOPMENT REPOSITORY. INTERFACES WILL VERY LIKELY CHANGE.

This is a toolkit for ambisonic speech research implemented in pytorch, meant to be implemented into collaborator's repositories as a subtree.
This toolkit includes:

- A **baseline ambisonic source separation model** from [A Dilated U-Net based approach...](https://ieeexplore.ieee.org/document/9287478).
- Implementations of various **ambisonic beamformers**.
- A dataset and a datamodule for **Spatial Librispeech** merged with the clean speech from **Librispeech**.
- Easy implementations of **spherical harmonic coefficients**.

## Necessary packages

This code relies on a few packages. You can install them into your existing conda environment with:

```bash
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install lightning pandas pyarrow fastparquet librosa matplotlib
```

## Installation as a subtree

1. Copy the Makefile found at the root of this repo into the root of your repo. This makefile has 3 useful commands to manage the subtree.
2. Run this to add the repository as a subtree.

```bash
make add-shared
```

You should have the kit added as a directory.

If you don't have make, you can download it:

```bash
# Linux
sudo apt-get install build-essential

# Windows
choco install make
```

## Interfacing with the ambispeechkit repository from your repository

As a subtree, your commits are saved in your own repository's commit history.
But you can also interface with the original public repository using these commands:

```bash
# pull latest from public ambispeechkit repo
make pull-shared

# push your latest commits to the public ambispeechkit repo
make push-shared
```

## Downloading and setting up spatial librispeech

Installation is easily done via the corresponding datamodule's **.prepare_data** method.
Even easier, just run the script with the corrects args from your root dir:

```bash
# Download a subset of sls
python -m ambispeechkit.datamodules.spatial_librispeech --train_head 100 --test_head 100 --reset_local_sls_metadata

# Download entire "lite" subset of sls
python -m ambispeechkit.datamodules.spatial_librispeech --download_lite_sls --reset_local_sls_metadata
```

Spatial librispeech is downloaded as .flac files.
I use ffmpeg to convert the flac files to wav.
If you don't have it yet, you can download it:

```bash
# Linux
sudo apt install ffmpeg

# Windows
choco install ffmpeg -y
```

If you stop the installation, re-running the script will just pick up from where it left off.

Don't forget to update your personal .gitignore with "datasets/*".
