# Overnight autoencoder speedrun

The first night I let Claude run experiments on its own: 30 ten-minute from-scratch runs to get the lowest test error on an STL-10 autoencoder, one change at a time, each logged and committed.

Part of a series of repos from my 2026 machine-learning sabbatical: a designer learning ML by building models from scratch and running experiments.

## What's here

- `train.py`: the training script extracted from my notebook (`make_train.py`, `stl10_jrope.py`)
- `LOG.md`: every run, its change and its score
- `RESEARCH.md`: the goal and rules for the night

## Running it

**Data:** set `ML_DATASETS` to the folder that holds the datasets (torchvision downloads STL-10 / MNIST there). Paths like `<home>` and `<data-root>` are placeholders for my machines. Each run: `python train.py` with settings as environment variables (see `LOG.md`).

## Results

Test MSE 0.0120 → 0.00445 (learning rate, β2 = 0.95, fewer blocks, CUDA Graphs). The next morning I found the low score came from a collapsed latent: the run optimized exactly the number it was given. Every later experiment logged latent health as well.

**Code:** written and run by Claude (Anthropic) as a coding agent, to my experiment specs. I chose the questions, the controls and the pass/fail rules, and read the results.

**Requirements:** Python 3.11+, PyTorch 2.x, torchvision, matplotlib, a CUDA GPU for anything beyond MNIST.
