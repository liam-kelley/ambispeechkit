# ambispeechkit

A toolkit for ambisonic speech research implemented in pytorch, meant to be implemented into collaborator's repositories as a subtree.

This toolkit includes:

- A baseline ambisonic speech enhancement / ambisonic source separation model from "A Dilated U-Net based approach for Multichannel Speech Enhancement from First-Order Ambisonics Recordings".
- Implementations of various beamformers (max_di, max_re, lc, mvdr, souden_mvdr, lcmv, max_sisnr, gevd_mwf).
- A dataset and a datamodule for Spatial Librispeech, merged with clean speech from Librispeech.
- Easy implementation of spherical harmonic coefficients.

## Installation

1. Copy the Makefile found at the root of this repo into the root of your repo. This makefile has 3 useful commands to manage the subtree.
2. Run this:

```bash
make add-shared
```

You should have the kit added as a directory.

If you don't have make, get it on linux with:

```bash
sudo apt-get install build-essential
```

And get it on windows with:

```bash
choco install make
```

3. This code relies on a few packages. You can install them into your conda environment with :

```bash
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install lightning pandas pyarrow fastparquet librosa matplotlib
```

You might also need ffmpeg if you convert the flac files to wav.
Idk about linux but on windows you can use:

```bash
choco install ffmpeg -y
```

## Interfacing with the ambispeechkit repository from your repository

As a subtree, your commits are saved in your own repository's commit history.
But you can also pull from the original public repository using this command :

```bash
make pull-shared
```

And push to the original public repository using this command :

```bash
make push-shared
```
