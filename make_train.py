#!/usr/bin/env python3
"""One-shot: build train.py from the notebook dump (cells 0-50, the pipeline
through the training loop; cell 51+ is post-hoc analysis and stays out)."""
import pathlib

HEADER = '''\
# train.py — runnable extraction of STL10-AE-Baseline-jrope-sparse-middle.ipynb
# (cells 0-50: setup, model, run management, training loop). Post-hoc analysis
# cells live in stl10_jrope.py (the full notebook dump, reference only).
#
# Headless defaults; override via env:
#   ML_DATASETS    (default <data-root>/caches/ml-datasets; STL10 auto-downloads)
#   ML_CHECKPOINTS (default ./artifacts/checkpoints)
#   EPOCHS         (default 300)
import os
import pathlib as _pl

import matplotlib
matplotlib.use("Agg")  # headless: plt.show() becomes a no-op

os.environ.setdefault("ML_DATASETS", "<data-root>/caches/ml-datasets")
os.environ.setdefault(
    "ML_CHECKPOINTS", str(_pl.Path(__file__).parent / "artifacts/checkpoints")
)
_pl.Path(os.environ["ML_DATASETS"]).mkdir(parents=True, exist_ok=True)

'''

src = pathlib.Path("stl10_jrope.py").read_text().splitlines(keepends=True)
body = "".join(src[:3375])  # through end of cell 50 (training loop)

body = body.replace(
    'split="train+unlabeled",\n    download=False, #already downloaded',
    'split="train+unlabeled",\n    download=True,'
)
body = body.replace(
    'split="test",\n    download=False, #already downloaded',
    'split="test",\n    download=True,'
)
body = body.replace(
    "num_epochs = 300  # ends at about 20.04 total epochs",
    'num_epochs = int(os.environ.get("EPOCHS", 300))',
)

pathlib.Path("train.py").write_text(HEADER + body)
print("wrote train.py", len((HEADER + body).splitlines()), "lines")
