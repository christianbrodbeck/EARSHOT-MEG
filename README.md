# Readme

This repository contains code for: 

Brodbeck, C., Hannagan, T., & Magnuson, J.S. (2025). Recurrent neural networks as neuro-computational models of human speech recognition. 


## Contents

- `earshot` directory contains an installable module for estimating EARSHOT models
- `EARSHOT-Burgundy` contains scripts and notebooks (see https://jupytext.readthedocs.io) used for generating the figures in the paper
  - [FIG - paper.py](EARSHOT-Burgundy/FIG%20-%20paper.py) code for most components for the figures included in the paper
  - [A - analyze embedding.py](EARSHOT-Burgundy/A%20-%20analyze%20embedding.py) and [A - Cohort activation.py](EARSHOT-Burgundy/A%20-%20Cohort%20activation.py): additional figure components
  - [make predictors.py](EARSHOT-Burgundy/make%20predictors.py) and [make_cluster_predictors.py](EARSHOT-Burgundy/make_cluster_predictors.py): create predictors for MEG data
- `examples` contains a demo notebook ([demo.py](examples/demo.py)) that runs a pretrained model on a stimulus and plots the results (see [Demo](#demo))
- `stimuli` contains a script used to generate the model input representations
- `tests` contains tests for certain functions in `earshot`


## Installation/Environment

Models were estimated using the environment in [env.yml](env.yml).

To set up this environment and install EARSHOT in it, use::

  % cd Code/EARSHOT-MEG        
  % mamba env create --file env.yml 
  % mamba activate earshot
  % pip install -e .

Installing EARSHOT compiles a Cython extension, which requires a C compiler.
On macOS, use the Xcode Command Line Tools (`xcode-select --install`); on Linux, use the system's `gcc`.

## Data

Spectrograms and trained models are not part of this repository.
`earshot` looks for them in a data directory, by default `~/Data/EARSHOT`.
To use a different location, set the `EARSHOT_DATA_DIR` environment variable, or call `earshot.set_data_dir(path)` before creating a `Model` or `Lexicon`.
The data directory is organized as follows::

  EARSHOT/
  ├── GAMMATONE_64_100/         (spectrograms: <Speaker>/<WORD>_<SPEAKER>.pickle)
  ├── Models/                   (one directory per trained model)
  └── Speakers.txt              (optional; defaults to the speakers used in the paper)

## Demo

[examples/demo.py](examples/demo.py) is a [marimo](https://marimo.io) notebook which creates a stimulus by concatenating words, runs it through a pretrained model, and plots the input, output and hidden layer activity.
With the data in place (see [Data](#data)), open it with::

  % marimo edit examples/demo.py

## Model input

The models are trained with gammatone spectrograms.
To create inputs for the model from wave files, use::

  import eelbrain
  
  wav = eelbrain.load.wav(path)
  gt = eelbrain.gammatone_bank(wav, 50, 10000, n=64, tstep=0.01)
