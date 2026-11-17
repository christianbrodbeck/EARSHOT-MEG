# Readme

This repository contains code for: 

Brodbeck, C., Hannagan, T., & Magnuson, J.S. (2025). Recurrent neural networks as neuro-computational models of human speech recognition. 


## Contents

- `earshot` directory contains an installable module for estimating EARSHOT models
- `EARSHOT-Burgundy` contains scripts and notebooks (see https://jupytext.readthedocs.io) used for generating the figures in the paper
  - [FIG - paper.py](EARSHOT-Burgundy/FIG%20-%20paper.py) code for most components for the figures included in the paper
  - [A - analyze embedding.py](EARSHOT-Burgundy/A%20-%20analyze%20embedding.py) and [A - Cohort activation.py](EARSHOT-Burgundy/A%20-%20Cohort%20activation.py): additional figure components
  - [make predictors.py](EARSHOT-Burgundy/make%20predictors.py) and [make_cluster_predictors.py](EARSHOT-Burgundy/make_cluster_predictors.py): create predictors for MEG data
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

## Model input

The models are trained with gammatone spectrograms.
To create inputs for the model from wave files, use::

  import eelbrain
  
  wav = eelbrain.load.wav(path)
  gt = eelbrain.gammatone_bank(wav, 50, 10000, n=64, tstep=0.01)
