# Research goal

Every experiment is a **10-minute, from-scratch training run** (fresh init,
no warm start from any previous checkpoint). Get the **clean reconstruction
loss** (`clean_term` in training metrics; `train clean` and `test clean` in
`evaluate_multiview`) **as low as possible within those 10 minutes.**

Two phases, in order:

1. **Until 0.0002 is reached:** minimize train-clean and test-clean at the
   10-minute mark. **The primary objective is wall-clock acceleration of
   clean reconstruction.** Any change is fair game — dropping or zeroing
   other loss terms, changing model size in either direction (larger is
   allowed if it converges faster in wall clock), batch size, lr/schedule,
   data, precision — but log exactly what was removed or changed so it can be
   restored later.
2. **Once a config hits ≤ 0.0002 in 10 minutes:** the priority flips to
   *adding back* what was removed — the other loss terms (masked, view,
   swd/bwd/c_swd/t_swd), original weight sizes / model width, full data, etc.
   — while still hitting ≤ 0.0002 within 10 minutes. Maximize how much of the
   original setup is restored subject to that constraint.

Codebase: STL10 masked autoencoder with jRoPE + sparse-middle transformer.
`train.py` is the runnable pipeline (notebook cells 0–50); `stl10_jrope.py`
is the full notebook dump including diagnostics — reference, don't execute.

## Baseline (unmodified train.py, 2026-09-18)

1 epoch = 1,640 steps ≈ 2.64 min (~47 ms/step, ~1,355 img/s after compile
warm-up). test clean 0.0917 → 0.0707 after one epoch. So a 10-minute box is
roughly 3.5 epochs of the baseline config.

## Owner findings (read before planning — these override the log's guesses)

- **The ~0.0045 test-clean floor from the 2026-09-18 overnight run (exp01–30)
  was latent collapse, not a bottleneck capacity limit.** The log's repeated
  "27x latent bottleneck caps MSE" hypothesis is wrong.
- **Cause: high lr combined with a high view-alignment term weight** drives
  z into collapse. Treat `lr` and `view_weight` as a coupled pair — do not
  raise one without checking the other, and do not read a flattening loss
  curve as a capacity ceiling until collapse has been ruled out.
- **Required diagnostic for every run:** log latent health at eval time —
  per-channel std of z, effective rank (e.g. fraction of singular values
  above 1% of the top one), and the fraction of dead channels. A run whose
  clean loss plateaus with shrinking z std / rank is collapsed; report it as
  such and treat the plateau value as meaningless for comparisons.
- Prior lr "brackets" in the log (2e-3 best, 4e-3 dead) were measured under
  collapse dynamics and should be re-bracketed once collapse is controlled.

## Success criteria

- Phase 1: a from-scratch run whose `train clean` and `test clean` both reach
  ≤ 0.0002 within 10 minutes of wall clock.
- Phase 2: the config that restores the most of the original setup while
  still meeting the phase-1 criterion.
- `LOG.md` records, for every attempt: best train/test clean at the 10-minute
  mark, the exact command, the diff made, and what is currently removed vs.
  restored relative to the original notebook.
- The winning config is reproducible from the repo (committed code + logged
  hyperparameters).

## Constraints

- **10-minute time-box per experiment, enforced mechanically** (e.g.
  `timeout 600 docker ... python train.py`, or a wall-clock check in the
  training loop that stops and runs the final eval). Never launch anything you
  can't see finish inside the box.
- **No warm starts.** Every run initializes from scratch. Do not load
  `checkpoint.pt` or any prior weights.
- One change per experiment so attribution stays clean; keep each diff small.
- GPU jobs run in the NGC container (see README for the exact command).
  Check the GPU is free (`nvidia-smi`) before launching — the owner also
  uses it interactively.
- Datasets: `<data-root>/caches/ml-datasets` (mount as `/data`,
  `ML_DATASETS=/data`). Checkpoints/plots land in `artifacts/` and
  `training_runs/` (both gitignored).
