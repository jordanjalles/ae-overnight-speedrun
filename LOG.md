# Research log

Conventions: every run is from scratch, `TIME_BUDGET_SEC=600` measured from the
first training step (torch.compile warm-up included, initial eval excluded),
with `timeout 900 docker ...` as the outer safety net. "train/test clean" are
`evaluate_multiview` (20 eval batches, dense eval mode, no RoPE jitter);
"step clean" is the training-loop `clean_term` average over the last 500-step
interval. Launch helper: `./run_exp.sh <name> KEY=VAL ...` (logs land in
`artifacts/logs/<name>.log`, run dirs in `training_runs/`).

## 2026-09-18 — iteration 1: time-box infra + exp01 pure reconstruction

**Infra (committed):** `train.py` hyperparameter cell now reads env overrides
(`RUN_NAME MASKED_VIEW VIEW_MODE LR LAP_W SWD_W CSWD_W TSWD_W BWD_W VIEW_W
SPARSE_MIDDLE TIME_BUDGET_SEC`); the training loop stops at the wall-clock
budget, runs the final `evaluate_multiview`, and prints a `TIME BUDGET ...
FINAL train clean X | test clean Y` line. Unset env = notebook defaults.

### exp01 — pure clean reconstruction (everything else removed)

Command:
```
./run_exp.sh exp01_pure_recon MASKED_VIEW=0 VIEW_MODE=none LAP_W=0 SWD_W=0 \
  CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 TIME_BUDGET_SEC=600
```
Removed vs. original notebook: masked view (masked recon term), view-alignment
term, laplacian term, swd/bwd/c_swd/t_swd terms. Kept: model (7.48M params),
data (train+unlabeled, 105k imgs), bs 64, lr 2e-3, sparse middle (keep 0.25),
RoPE jitter, LR spike guard.

Result at the 10-min mark (13,773 steps, 8.4 epochs, ~36 ms/step, ~1,800 img/s):

| metric | value |
|---|---|
| FINAL train clean | 0.09745 |
| FINAL test clean | 0.09758 |
| best test clean (epoch end) | 0.07009 @E2 |
| step clean (last 500 steps) | 0.04298 |

Per-epoch test clean: E1 0.0711, E2 0.0701, E3 0.1134, E4 0.1294, E5 0.1191,
E6 0.1069, E7 0.0827, E8 0.0835, E9(partial) 0.0976. LR guard fired at every
epoch end from step 6,560; scale 1.0 each time except step 13,120 (0.5 →
lr 1e-3).

Implications:
- Dropping the six other terms gave a ~1.3x speedup per step (47 → 36 ms) but
  the clean loss is nowhere near 2e-4: the training-step MSE is 0.043 after
  13.7k steps (pixel variance is ~0.07, so the model barely reconstructs).
- **Train/eval mismatch dominates:** step clean 0.043 vs. eval train clean
  0.097 on the same data. Eval runs dense (all tokens through all 8 blocks,
  `sparse_middle` only active in `.training`) and without RoPE scale jitter.
  The eval numbers also oscillate wildly (0.070 → 0.129 → 0.083) while the
  training loss decreases monotonically, consistent with the dense path
  being a distribution the weights were never trained on.
- Next: exp02 = same config with `SPARSE_MIDDLE=0` (train dense like eval).
  If that closes the gap, the following suspects are RoPE jitter, the
  1e-4 layer-scale init / lr, and the 1,024-dim latent bottleneck
  (32 tokens × 32 ch for a 27,648-dim image, 27x compression), which likely
  caps achievable MSE far above 2e-4 regardless of optimization.

## 2026-09-18 — iteration 1 (re-run after kill): exp02 partial, exp03 lr

The previous iteration was killed by the 30-min wall clock while exp02 was
still running (no FINAL line). Its partial log is still informative.

### exp02 — pure recon, dense middle (SPARSE_MIDDLE=0), killed at ~E4

Command:
```
./run_exp.sh exp02_pure_recon_dense MASKED_VIEW=0 VIEW_MODE=none LAP_W=0 \
  SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 TIME_BUDGET_SEC=600
```
Removed vs. original: same six terms as exp01, plus sparse middle (dense
training). 46 ms/step, ~1,390 img/s (vs 36 ms sparse): dense costs 1.3x
per step.

| point | step clean (500-step avgs) | test clean |
|---|---|---|
| start | — | 0.09166 |
| E1 (1,640 steps) | 0.0682 / 0.0693 / 0.0692 / 0.0701 | 0.06829 |
| E2 (3,280) | 0.0698 / 0.0700 / 0.0698 / 0.0704 | 0.06846 |
| E3 (4,920) | 0.0696 / 0.0697 / 0.0700 / 0.0696 | 0.06833 |
| E4 (partial, 6,420) | 0.0696 / 0.0698 / 0.0697 | killed |

LR probe at step 6,560 kept lr 2e-3: all smaller scales gave loss ratio
0.998–1.000, i.e. the loss is insensitive to lr (dead model).

Implications:
- Dense training closes the train/eval gap (step clean 0.070 ≈ test clean
  0.068), confirming exp01's gap was the sparse-vs-dense mismatch.
- But the loss is flat at ~0.069 from step 500 onward: the model collapses
  to a near-constant / mean-color predictor within the first 500 steps and
  never recovers. Sparse (exp01) reached 0.043 step-clean, so the sparse
  path was the healthier optimization, plausibly because fewer tokens per
  step means smaller effective updates.
- Hypothesis for exp03: lr 2e-3 AdamW with no warmup (layer-scale 1e-4 is
  inflated to O(1e-2) by Adam within ~10 steps regardless of gradient size)
  blows the transformer up early. Test LR=5e-4, one change.

### exp03 — exp02 + LR=5e-4 (dense, pure recon)

Command:
```
./run_exp.sh exp03_dense_lr5e-4 MASKED_VIEW=0 VIEW_MODE=none LAP_W=0 SWD_W=0 \
  CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=5e-4 TIME_BUDGET_SEC=600
```
No code diff; one change vs. exp02: lr 2e-3 → 5e-4. Removed vs. original
notebook: masked view, view term, laplacian, swd/bwd/c_swd/t_swd, sparse
middle; lr changed. Kept: 7.48M model, bs 64, full data, RoPE jitter, LR
guard (never fired a reduction).

Result at the 10-min mark (11,167 steps, 6.8 epochs, ~47 ms/step):

| metric | value |
|---|---|
| FINAL train clean | 0.01200 |
| FINAL test clean | 0.01203 |
| step clean (last interval) | 0.01151 |

Per-epoch test clean: E1 0.02444, E2 0.02118, E3 0.01853, E4 0.01626,
E5 0.01393, E6 0.01220, E7(partial) 0.01203. Step clean at 500/1000/1500:
0.0605 / 0.0337 / 0.0287 (exp02 at lr 2e-3: 0.068 / 0.069 / 0.069).

Implications:
- **lr 2e-3 was the blocker.** At 5e-4 the same model learns immediately and
  is 5.7x better than exp02 at the 10-min mark (0.0120 vs 0.068), and still
  improving ~12%/epoch with no sign of plateau. Train ≈ test, so no
  overfitting; we are optimization-limited, not data-limited.
- Still 60x above the 0.0002 target. At ~12%/epoch the current trajectory
  would not get there in any reasonable wall clock, so bigger levers are
  needed: (a) lr sweep around 5e-4 (1e-3 and 2.5e-4) plus a warmup+cosine
  schedule so a higher peak lr is safe; (b) the 1,024-dim latent bottleneck
  (27x compression) almost certainly caps MSE well above 2e-4 — test by
  widening z (emb_dim 32→128 or n_latent_tokens 32→64) as a model-size
  change, logged for restoration in phase 2; (c) larger batch / bf16 already
  on; (d) the sigmoid output + MSE saturates gradients near 0/1.
- Next iteration: exp04 = exp03 + LR=1e-3 (bracket the lr), then latent
  width. Baseline for the loop is now exp03.

## 2026-09-18 — iteration 2: exp04 lr bracket (1e-3)

**Infra (committed, no behaviour change when unset):** `train.py` override
cell now also reads `EMB_DIM` and `N_LATENT_TOKENS` (z bottleneck size) so
the latent-width experiment needs no code diff. `run_exp.sh` unchanged.

### exp04 — exp03 + LR=1e-3 (dense, pure recon)

Command:
```
./run_exp.sh exp04_dense_lr1e-3 MASKED_VIEW=0 VIEW_MODE=none LAP_W=0 SWD_W=0 \
  CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=1e-3 TIME_BUDGET_SEC=600
```
One change vs. exp03: lr 5e-4 → 1e-3. Removed vs. original notebook: masked
view, view term, laplacian, swd/bwd/c_swd/t_swd, sparse middle; lr changed
(2e-3 → 1e-3). Kept: 7.48M model, z = 32 tokens × 32 ch, bs 64, full data,
RoPE jitter, LR guard.

Result at the 10-min mark (11,132 steps, 6.8 epochs, ~47 ms/step, ~1,375 img/s):

| metric | exp04 (lr 1e-3) | exp03 (lr 5e-4) |
|---|---|---|
| FINAL train clean | **0.00912** | 0.01200 |
| FINAL test clean | **0.00909** | 0.01203 |
| step clean (last interval) | 0.00923 | 0.01151 |

Per-epoch test clean, exp04 vs exp03: E1 0.02416/0.02444, E2 0.01941/0.02118,
E3 0.01426/0.01853, E4 0.01174/0.01626, E5 0.01024/0.01393, E6 0.00979/0.01220,
E7 0.00909/0.01203. Step clean at 500/1000/1500: 0.0581 / 0.0343 / 0.0279
(exp03: 0.0605 / 0.0337 / 0.0287).

LR guard: at step 6,560 (E4 end) it halved lr to 5e-4 (probe ratios 0.125×
0.994, 0.25× 0.993, 0.5× 1.002, 0.75× 1.027, 1× 1.067 — i.e. lr 1e-3 was
becoming unstable by E4); kept 5e-4 at 8,200 and 9,840. So exp04 is
effectively "1e-3 for 4 epochs then 5e-4", a crude step schedule.

Implications:
- lr 1e-3 is 24% better than 5e-4 at the 10-min mark (0.0091 vs 0.0120) and
  the gap widens each epoch until the guard halves the lr; the halving itself
  gave +12.8% at E5. Early steps (≤1,500) are lr-insensitive between 5e-4 and
  1e-3, so the gain comes from mid-run, not warm-up.
- The guard's probe at E4 shows the loss surface prefers *lower* lr late,
  which argues for an explicit warmup + cosine/linear decay to 0 over the
  600 s budget (peak 1e-3 to 2e-3) instead of the epoch-granular guard.
  Decay-to-zero typically buys a large final-loss drop in short-budget runs.
- Still 45x above 0.0002 and improving only ~7%/epoch at the end, so the
  schedule alone will not close it; the 27x latent compression is the next
  suspect. `EMB_DIM`/`N_LATENT_TOKENS` knobs are now in place.
- Next iteration (pick one, one change each): exp05 = exp04 + time-based
  linear-warmup (~300 steps) + cosine decay to 0 at TIME_BUDGET_SEC, peak
  1e-3 (needs a small code change in the training loop); exp06 = exp04 +
  EMB_DIM=128 (z 32×128, 6.75x compression) to test the bottleneck ceiling.
  Baseline for the loop is now exp04.

## 2026-09-18 — iteration 3: exp05 warm-up + cosine-to-zero schedule

**Infra (committed, no behaviour change when unset):** `train.py` training
loop now supports `LR_SCHED=cosine` (linear warm-up over `WARMUP_STEPS`
steps, then cosine decay from `LR` to `LR*LR_MIN_FRAC` at `TIME_BUDGET_SEC`,
wall-clock based). LR-guard reductions are preserved as a multiplier on the
scheduled value. Unset = constant lr as before.

### exp05 — exp04 + LR_SCHED=cosine (warm-up 300 steps, decay to 0 at 600 s)

Command:
```
./run_exp.sh exp05_dense_lr1e-3_cosine MASKED_VIEW=0 VIEW_MODE=none LAP_W=0 \
  SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=1e-3 \
  LR_SCHED=cosine WARMUP_STEPS=300 LR_MIN_FRAC=0 TIME_BUDGET_SEC=600
```
One change vs. exp04: constant lr 1e-3 (+ guard halving at E4) → warm-up +
cosine decay to 0 over the 600 s budget, peak 1e-3. Removed vs. original
notebook: masked view, view term, laplacian, swd/bwd/c_swd/t_swd, sparse
middle; lr 2e-3 → 1e-3 peak with schedule. Kept: 7.48M model, z 32×32,
bs 64, full data, RoPE jitter, LR guard (probes at E4/E5/E6 all kept
scale 1).

Result at the 10-min mark (11,095 steps, 6.8 epochs, ~47 ms/step):

| metric | exp05 (cosine→0) | exp04 (const 1e-3, guard) |
|---|---|---|
| FINAL train clean | 0.01190 | **0.00912** |
| FINAL test clean | 0.01191 | **0.00909** |

Per-epoch test clean, exp05 vs exp04: E1 0.02845/0.02416, E2 0.01977/0.01941,
E3 0.01533/0.01426, E4 0.01344/0.01174, E5 0.01235/0.01024, E6 0.01196/0.00979,
E7 0.01191/0.00909. Scheduled lr at the E4/E5/E6 probes: 3.5e-4 / 1.6e-4 /
3.2e-5; gain per epoch collapsed to +3.1% (E6) and +0.4% (E7) as lr → 0.

Implications:
- **Cosine-to-zero over the full budget is 31% worse than constant lr.**
  exp05 is behind exp04 at every epoch: the 300-step warm-up costs ~0.004 at
  E1 (early steps were lr-insensitive, so the warm-up only delays), and from
  E3 on the decaying lr simply makes less progress per epoch. The run is
  progress-limited (loss still falling steadily at fixed lr), not
  noise-limited, so lowering lr early buys nothing. exp04's E4 halving gave
  +12.8% only because lr 1e-3 had actually become unstable there (probe
  ratio 1.067 at 1×), which is a different regime from a pre-planned decay.
- Schedule conclusion: keep lr high as long as it is stable; at most a short
  decay in the last ~10–15% of the budget. Do not spend more iterations on
  schedule shape — the remaining 45x gap to 0.0002 will not come from it.
- Next iteration: exp06 = exp04 + `EMB_DIM=128` (z 32×128, 6.75x compression
  instead of 27x) to test the latent-bottleneck ceiling; then, if that helps,
  `N_LATENT_TOKENS=64`. Baseline for the loop remains exp04
  (test clean 0.00909).

## 2026-09-18 — iteration 5: exp06 latent width (EMB_DIM=128), exp07 batch 128

exp06 was launched by the (killed) iteration 4 and finished during this one;
its container outlived the killed client, so the output was recovered via
`docker logs -f` into `artifacts/logs/exp06_dense_lr1e-3_emb128.docker.log`
(the `.log` file only has the container banner).

### exp06 — exp04 + EMB_DIM=128 (z 32 tokens × 128 ch, 6.75x compression)

Command:
```
./run_exp.sh exp06_dense_lr1e-3_emb128 MASKED_VIEW=0 VIEW_MODE=none LAP_W=0 \
  SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=1e-3 \
  EMB_DIM=128 TIME_BUDGET_SEC=600
```
One change vs. exp04: emb_dim 32 → 128 (model-size change). Removed vs.
original notebook: masked view, view term, laplacian, swd/bwd/c_swd/t_swd,
sparse middle; lr 2e-3 → 1e-3; z width 32 → 128.

Result at the 10-min mark (11,001 steps, 6.7 epochs, ~47 ms/step, ~1,355 img/s):

| metric | exp06 (emb 128) | exp04 (emb 32) |
|---|---|---|
| FINAL train clean | 0.01146 | **0.00912** |
| FINAL test clean | 0.01147 | **0.00909** |

Per-epoch test clean, exp06 vs exp04: E1 0.02586/0.02416, E2 0.02210/0.01941,
E3 0.01814/0.01426, E4 0.01417/0.01174, E5 0.01251/0.01024, E6 0.01179/0.00979,
E7 0.01147/0.00909. Step clean at 500/1000/1500: 0.0556 / 0.0379 / 0.0312
(exp04: 0.0581 / 0.0343 / 0.0279). LR guard halved lr to 5e-4 at E4 (same as
exp04; probe 1× ratio 1.006), kept 5e-4 at E5/E6.

Implications:
- **The 27x latent bottleneck is not the limiter at this loss level.** A 4x
  wider z is 26% *worse* at 10 min and behind exp04 at every epoch from E2 on,
  with identical ms/step. Whatever caps progress (~0.009 after 7 epochs, still
  falling ~5–7%/epoch) is optimization speed, not representational capacity.
  Do not spend more iterations on latent width until the loss is far lower.
- Remaining big levers for wall-clock: throughput (bs 64 at 47 ms/step on a
  7.5M-param model is launch/overhead-bound, so a larger batch should be
  nearly free per step and double images/s), and the LR guard's E4 halving
  shows lr 1e-3 goes unstable mid-run — larger batch also stabilizes it.

### exp07 — exp04 + BATCH_SIZE=128 (bs 64 → 128, lr 1e-3 unchanged)

**Infra (committed, no behaviour change when unset):** `train.py` train
DataLoader reads `BATCH_SIZE` (default 64); `run_exp.sh` header lists it.

Command:
```
./run_exp.sh exp07_dense_lr1e-3_bs128 MASKED_VIEW=0 VIEW_MODE=none LAP_W=0 \
  SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=1e-3 \
  BATCH_SIZE=128 TIME_BUDGET_SEC=600
```
One change vs. exp04: batch 64 → 128. Removed vs. original notebook: masked
view, view term, laplacian, swd/bwd/c_swd/t_swd, sparse middle; lr 2e-3 →
1e-3; bs 64 → 128. Kept: 7.48M model, z 32×32, full data, RoPE jitter, guard.

Result at the 10-min mark (5,308 steps, 6.5 epochs, ~97 ms/step, ~1,315 img/s):

| metric | exp07 (bs 128) | exp04 (bs 64) |
|---|---|---|
| FINAL train clean | 0.01034 | **0.00912** |
| FINAL test clean | 0.01038 | **0.00909** |
| ms/step, img/s | 97, 1,315 | 47, 1,375 |

Per-epoch test clean, exp07 vs exp04: E1 0.02760/0.02416, E2 0.02052/0.01941,
E3 0.01673/0.01426, E4 0.01324/0.01174, E5 0.01238/0.01024, E6 0.01094/0.00979,
E7 0.01038/0.00909. LR guard never fired a reduction (lr 1e-3 stayed stable
all run, unlike bs 64 which needed a halving at E4).

Implications:
- **The GPU is compute-bound already at bs 64:** doubling the batch doubles
  ms/step (47 → 97) with the same img/s, so throughput cannot be bought with
  batch size. Half as many optimizer steps at the same lr = 14% worse at 10
  min. (bs 128 did make lr 1e-3 stable through E7, so if we ever need higher
  lr, larger batch is the enabler — but not a win by itself.)
- Together with exp06 (wider z: no help) and exp05 (schedule: no help), the
  levers tried so far all say the same thing: we are limited by *optimizer
  steps × per-step progress* at ~1,350 img/s. The remaining wall-clock levers
  are therefore per-step cost of the *model* (fewer/cheaper blocks, e.g.
  restoring `SPARSE_MIDDLE=1` now that lr is sane — exp01 showed sparse runs
  at 36 ms/step, 1.3x more steps — or a smaller/shallower encoder-decoder)
  and per-step progress (optimizer: try lr 1e-3 with bs 64 but a lower
  weight-decay / beta2, or Muon-style updates), plus checking whether the
  47 ms/step is dominated by something silly (dataloader without workers,
  fp32 eval, RoPE jitter tables rebuilt every step) via a quick profile.
- Next iteration: exp08 = exp04 + `SPARSE_MIDDLE=1` (train sparse at lr 1e-3;
  train/eval mismatch was the exp01 problem, but at lr 1e-3 the model may be
  robust enough and gets 1.3x the steps). If the mismatch persists, follow
  with a 1-epoch profile of the 47 ms step. Baseline remains exp04
  (test clean 0.00909).

## 2026-09-18 — iteration 6: exp08 sparse middle at lr 1e-3, dataloader-worker probe

### exp08 — exp04 + SPARSE_MIDDLE=1 (sparse middle restored, lr 1e-3)

Command:
```
./run_exp.sh exp08_sparse_lr1e-3 MASKED_VIEW=0 VIEW_MODE=none LAP_W=0 SWD_W=0 \
  CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=1 LR=1e-3 TIME_BUDGET_SEC=600
```
No code diff; one change vs. exp04: sparse middle (keep 0.25) back on during
training. Removed vs. original notebook: masked view, view term, laplacian,
swd/bwd/c_swd/t_swd; lr 2e-3 → 1e-3. Restored relative to exp04: sparse
middle. Kept: 7.48M model, z 32×32, bs 64, full data, RoPE jitter, guard.

Result at the 10-min mark (14,129 steps, 8.6 epochs, ~35 ms/step, ~1,810 img/s):

| metric | exp08 (sparse) | exp04 (dense) |
|---|---|---|
| FINAL train clean | 0.01043 | **0.00912** |
| FINAL test clean | 0.01046 | **0.00909** |
| step clean (last interval) | 0.01087 | 0.00923 |
| steps in 600 s | 14,129 | 11,132 |

Per-epoch test clean, exp08 vs exp04: E1 0.02906/0.02416, E2 0.02445/0.01941,
E3 0.02055/0.01426, E4 0.01727/0.01174, E5 0.01569/0.01024, E6 0.01428/0.00979,
E7 0.01172/0.00909, E8 0.01079/—, E9(partial) 0.01046/—. LR guard kept
scale 1 at every probe (1× ratio 0.994–0.996): lr 1e-3 stays stable with
sparse training, unlike dense which needed a halving at E4.

Implications:
- **Sparse middle at lr 1e-3 is 15% worse than dense at 10 min.** 1.27x more
  steps (35 vs 47 ms/step) does not compensate for ~30% less progress per
  epoch (E7: 0.0117 vs 0.0091). The exp01 train/eval mismatch is gone at this
  lr (step clean 0.0109 ≈ eval 0.0104), so the issue is purely that dropping
  75% of tokens in the middle blocks gives a weaker gradient per step, not
  the dense-eval distribution shift.
- Sparse *is* the phase-2 restore target and it costs only 15%, so it can
  be restored cheaply later; for phase 1 stay dense. Baseline remains exp04
  (test clean 0.00909).
- All cheap knobs (lr, schedule, z width, batch, sparsity) are now bracketed
  and none move the 45x gap; the run is progress-per-step limited at
  ~1,350 img/s. Next levers: (1) confirm ms/step is not dataloader-bound
  (probe below), (2) optimizer changes (weight decay / betas, Muon), (3) a
  cheaper model (fewer blocks) to trade capacity for steps.

### probe_workers4 — dataloader workers (150 s throughput probe, not a 10-min result)

**Infra (committed, no behaviour change when unset):** train DataLoader reads
`NUM_WORKERS` (default 0 = notebook behaviour; >0 also sets `pin_memory` and
`persistent_workers`); `run_exp.sh` passes `--shm-size=1g` so worker
processes are not limited by the container's 64 MB default /dev/shm.

Command:
```
./run_exp.sh probe_workers4 MASKED_VIEW=0 VIEW_MODE=none LAP_W=0 SWD_W=0 \
  CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=1e-3 NUM_WORKERS=4 \
  TIME_BUDGET_SEC=150
```
exp04 config with 4 loader workers, 150 s budget (2,874 steps).

| step | probe (4 workers) ms/step, img/s, step clean | exp04 (0 workers) |
|---|---|---|
| 1000 | 39, 1,631, 0.0319 | 46, 1,384, 0.0343 |
| 1500 | 39, 1,639, 0.0271 | 46, 1,386, 0.0279 |
| 2874 | test clean 0.01851 @ 150 s | (exp04 E1 @ ~1.3 min: 0.02416) |

Implications:
- **The 47 ms step was ~15% dataloader-bound**, not purely compute-bound as
  concluded from exp07 (batch doubling scales the loader cost too, so it could
  not tell them apart). Workers give +18% img/s for free, and the loss at a
  given step is unchanged (same optimization, just faster).
- Next iteration: exp09 = exp04 + NUM_WORKERS=4 for the full 600 s (expect
  ~13k steps, test clean ≈ exp04's trajectory extended by ~1.2 epochs,
  ~0.0085). Then try NUM_WORKERS=8 to see whether 39 ms is now the GPU floor.
  Baseline remains exp04 (test clean 0.00909) until exp09 lands.

## 2026-09-18 — iteration 7: exp09 dataloader workers for the full 10 min

### exp09 — exp04 + NUM_WORKERS=4 (dense, pure recon, lr 1e-3)

Command:
```
./run_exp.sh exp09_dense_lr1e-3_workers4 MASKED_VIEW=0 VIEW_MODE=none LAP_W=0 \
  SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=1e-3 \
  NUM_WORKERS=4 TIME_BUDGET_SEC=600
```
No code diff; one change vs. exp04: 4 DataLoader workers (+ pin_memory,
persistent_workers) instead of 0. Removed vs. original notebook: masked view,
view term, laplacian, swd/bwd/c_swd/t_swd, sparse middle; lr 2e-3 → 1e-3.
Kept: 7.48M model, z 32×32, bs 64, full data, RoPE jitter, LR guard.
Workers are a pure throughput change (not part of the original setup's loss
definition), so nothing new needs restoring in phase 2.

Result at the 10-min mark (12,869 steps, 7.8 epochs, 39 ms/step, ~1,635 img/s):

| metric | exp09 (4 workers) | exp04 (0 workers) |
|---|---|---|
| FINAL train clean | **0.00743** | 0.00912 |
| FINAL test clean | **0.00741** | 0.00909 |
| step clean (last interval) | 0.00746 | 0.00923 |
| steps in 600 s | 12,869 | 11,132 |

Per-epoch test clean, exp09 vs exp04: E1 0.02228/0.02416, E2 0.01612/0.01941,
E3 0.01111/0.01426, E4 0.00970/0.01174, E5 0.00834/0.01024, E6 0.00789/0.00979,
E7 0.00763/0.00909, E8 0.00741/—. Step clean at 500/1000/1500: 0.0538 /
0.0319 / 0.0271 (exp04: 0.0581 / 0.0343 / 0.0279). Epoch wall time 1.07–1.14
min vs ~1.3 min.

LR guard: halved to 5e-4 at step 6,560 (E4; 1× ratio 1.068, same as exp04)
and halved *again* to 2.5e-4 at 8,200 (E5; 1× ratio 1.0032 — marginal, 0.5×
0.998), then kept 2.5e-4 at E6/E7. Per-epoch gain after the second halving
collapsed to +5.4% / +3.3% / +2.9% (E6–E8), vs +12.7% / +14.0% at E4–E5.

Implications:
- **New best: 0.00741, 18.5% better than exp04**, from a throughput-only
  change (+16% steps). The per-step trajectory is also slightly better early
  on (E1–E3), which is noise/seed plus the same steps landing earlier; there
  is no reason to expect a loss-per-step change from workers.
- exp09 is ahead of exp04 at every epoch *by more than the step count
  explains* (E3 0.0111 vs 0.0143 at nearly the same step), so run-to-run
  variance at fixed config is likely ±5–10% at the epoch level; single-run
  comparisons below ~10% should not be trusted.
- The second LR halving at E5 looks premature (1× ratio 1.003 is within
  noise) and the run then crawled at 2.5e-4 for the last 3 epochs. The guard's
  reduction threshold (ratio > 1.0 at 1×?) is now the most visible
  optimization inefficiency: exp07 (bs 128) showed lr 1e-3 can stay stable
  all run. Candidate next steps, one change each: (a) raise the guard's
  trigger threshold (e.g. only halve if 1× ratio > 1.02) or cap it at one
  halving per run; (b) `NUM_WORKERS=8` to check whether 39 ms is the GPU
  floor; (c) a cheaper model (fewer blocks) for more steps.
- Still 37x above 0.0002; the remaining gap will not come from throughput
  tweaks alone. Baseline for the loop is now **exp09 (test clean 0.00741,
  NUM_WORKERS=4 becomes a default knob for all future runs).**
- Next iteration: exp10 = exp09 + guard threshold change (a), since the
  E5 halving cost an estimated ~10% of final loss (E6–E8 gains ~3%/epoch at
  2.5e-4 vs ~13%/epoch at 5e-4).

## 2026-09-18 — iteration 8: exp10 LR-guard reduction margin (GUARD_GAIN)

**Infra (committed, no behaviour change when unset):** `_probe_smaller_lr`
in `train.py` reads `GUARD_GAIN` (default 0.995 = notebook): a probe scale
must beat the current-lr probe loss by this factor before lr is reduced.
`run_exp.sh` header lists it.

### exp10 — exp09 + GUARD_GAIN=0.98 (dense, pure recon, lr 1e-3, 4 workers)

Command:
```
./run_exp.sh exp10_dense_lr1e-3_workers4_guard098 MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=1e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 TIME_BUDGET_SEC=600
```
One change vs. exp09: guard reduction margin 0.5% → 2%. Removed vs. original
notebook: masked view, view term, laplacian, swd/bwd/c_swd/t_swd, sparse
middle; lr 2e-3 → 1e-3; guard margin 0.995 → 0.98. Kept: 7.48M model,
z 32×32, bs 64, full data, RoPE jitter, LR guard (probing unchanged).

Result at the 10-min mark (12,818 steps, 7.8 epochs, 39 ms/step, ~1,620 img/s):

| metric | exp10 (margin 2%) | exp09 (margin 0.5%) |
|---|---|---|
| FINAL train clean | **0.00718** | 0.00743 |
| FINAL test clean | **0.00715** | 0.00741 |
| step clean (last interval) | 0.00729 | 0.00746 |
| lr after E4 / E5 probe | 5e-4 / 5e-4 | 5e-4 / 2.5e-4 |

Per-epoch test clean, exp10 vs exp09: E1–E5 identical to 4 digits (0.02228,
0.01612, 0.01111, 0.00970, 0.00834), E6 **0.01165**/0.00789, E7 0.00743/0.00763,
E8 0.00715/0.00741. Probes: E4 identical to exp09 (halved to 5e-4); E5 0.5×
0.9980 vs 1× 1.0032 → kept 5e-4 (exp09 halved here); E6 1× 0.9186 (kept);
E7 1× 1.0001 (kept).

Step-clean 500-step averages, E6–E8, exp10 vs exp09: 0.00839/0.00828,
0.00813/0.00814, 0.00797/0.00802, [9840] 0.00859/0.00803, 0.00872/0.00794,
0.00775/0.00780, 0.00770/0.00777, 0.00748/0.00767, 0.00747/0.00762,
0.00731/0.00750, 0.00729/0.00746.

Implications:
- **New best: 0.00715, 3.5% better than exp09.** Keeping lr at 5e-4 instead
  of 2.5e-4 for E6–E8 gives ~5%/epoch at the end vs ~3%.
- **Runs are deterministic.** exp10 reproduced exp09 to 4 digits through E5
  (same seed, same worker count), so the "±5–10% run-to-run variance" claim
  in the exp09 entry is wrong; the exp09-vs-exp04 gap at equal step was a
  worker-count effect (different RNG stream / data order), not noise.
  Single-run A/B at matched config is trustworthy.
- **lr 5e-4 has a genuine instability at bs 64:** in the last ~140 steps of
  E6 the step loss jumped (0.00797 → 0.00859 interval avg; E6 eval 0.01165,
  +48% vs E5) and took ~1,000 steps to recover; the E6 probe's 1× 0.9186 is
  the recovery slope, not a "5e-4 is great" signal. So the guard's E5 halving
  in exp09 was marginal but not useless — the win from not halving (E7/E8
  ahead) only barely exceeded the spike cost. exp07 (bs 128) never spiked
  at lr 1e-3, so instability scales with per-step gradient noise.
- The gain is small and the E6 spike suggests the real fix is stability at
  higher lr rather than guard tuning. Next iteration candidates (one change
  each, on top of exp10): (a) gradient clipping (e.g. clip_grad_norm 1.0) so
  lr 1e-3 can survive E4+ without halving; (b) AdamW beta2 0.999 → 0.95 for
  the same reason; (c) GUARD_GAIN=1.0 with clipping (guard off) later.
  Baseline for the loop is now **exp10 (test clean 0.00715)**; NUM_WORKERS=4
  and GUARD_GAIN=0.98 are default knobs for future runs. Still 36x above
  0.0002.

## 2026-09-18 — iteration 9: exp11 gradient clipping (GRAD_CLIP)

**Infra (committed, no behaviour change when unset):** `train_step` in
`train.py` applies `torch.nn.utils.clip_grad_norm_` over all optimizer params
before `optimizer.step()` when `GRAD_CLIP` > 0 (default 0 = notebook, off).
The clip is skipped for the LR guard's `ScoreOnly` probe stub (no
`param_groups`) — the first launch of exp11 crashed at the E4 probe with
`AttributeError: 'ScoreOnly' object has no attribute 'param_groups'`
(log kept as `artifacts/logs/exp11_crashed_E4probe.log`; its E1–E3 evals
were 0.02228 / 0.01612 / 0.01111, identical to exp10). `run_exp.sh` header
lists `GRAD_CLIP`.

### exp11 — exp10 + GRAD_CLIP=1.0 (dense, pure recon, lr 1e-3, 4 workers, guard 0.98)

Command:
```
./run_exp.sh exp11_dense_lr1e-3_workers4_guard098_clip1 MASKED_VIEW=0 \
  VIEW_MODE=none LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 \
  SPARSE_MIDDLE=0 LR=1e-3 NUM_WORKERS=4 GUARD_GAIN=0.98 GRAD_CLIP=1.0 \
  TIME_BUDGET_SEC=600
```
One change vs. exp10: global grad-norm clip at 1.0. Removed vs. original
notebook: masked view, view term, laplacian, swd/bwd/c_swd/t_swd, sparse
middle; lr 2e-3 → 1e-3; guard margin 0.995 → 0.98; grad clip added. Kept:
7.48M model, z 32×32, bs 64, full data, RoPE jitter, LR guard.

Result at the 10-min mark (12,236 steps, 7.5 epochs, 40–51 ms/step, ~1,250–1,600 img/s):

| metric | exp11 (clip 1.0) | exp10 (no clip) |
|---|---|---|
| FINAL train clean | 0.00736 | **0.00718** |
| FINAL test clean | 0.00733 | **0.00715** |
| step clean (last interval) | 0.00732 | 0.00729 |
| steps in 600 s | 12,236 | 12,818 |

Per-epoch test clean, exp11 vs exp10: E1–E7 **identical to 4 digits**
(0.02228, 0.01612, 0.01111, 0.00970, 0.00834, 0.01165, 0.00743), E8
0.00733/0.00715 (exp11 reached E8 later, at step 12,236 vs ~12,800). All
four LR probes identical to exp10 (E4 1× 1.0678 → halve to 5e-4; E5 1×
1.0032 keep; E6 1× 0.9186 keep; E7 1× 1.0001 keep). The E6 loss spike
(eval 0.01165, +40% vs E5) reproduced exactly.

Implications:
- **GRAD_CLIP=1.0 is a no-op:** the global grad norm never exceeds 1.0 in
  this run (bit-identical trajectory through E7 including the E6 spike), so
  clipping at 1.0 neither prevents the E4 instability nor the E6 spike. The
  spike is therefore not a large-gradient event; it is a small-gradient
  drift at lr 5e-4 (Adam-normalised updates are ~lr-sized regardless of
  gradient magnitude), so clipping at any threshold above the typical norm
  cannot help and a much lower clip would just act like a lower lr.
- The 0.00733 vs 0.00715 difference is entirely throughput: this run was
  ~5% slower per step (40–51 ms vs 39; the first exp11 launch had crashed
  and its worker/compile caches were cold, or host contention), reaching
  E8 ~600 steps later. Step-clean at the last interval (0.00732 vs 0.00729)
  confirms equal loss-per-step. Do not count exp11 as a regression;
  baseline stays **exp10 (test clean 0.00715)**.
- Keep `GRAD_CLIP=0` as the default (clip is inert, but costs a norm
  reduction per step). Since clipping cannot stabilise lr 1e-3 past E4, the
  remaining stability lever is the optimizer's *update* magnitude, not the
  gradient: next iteration try **AdamW beta2 0.999 → 0.95** (faster
  second-moment tracking; standard fix for mid-run loss spikes in
  transformers) as exp12, one change on top of exp10, with a `BETA2` knob.
  If that also fails to hold lr 1e-3, move to the model-cost lever (fewer
  blocks) for more steps. Still 36x above 0.0002.

## 2026-09-18 — iteration 10: exp12 AdamW beta2 0.95 (BETA2)

**Infra (committed, no behaviour change when unset):** `train.py` reads
`BETA1` (default 0.9) and `BETA2` (default 0.999) and passes
`betas=(BETA1, BETA2)` to the `AdamW` in `init_models`. The LR guard's probe
reuses the live optimizer, so the betas apply there too. `run_exp.sh` header
lists both.

### exp12 — exp10 + BETA2=0.95 (dense, pure recon, lr 1e-3, 4 workers, guard 0.98)

Command:
```
./run_exp.sh exp12_dense_lr1e-3_workers4_guard098_beta2_095 MASKED_VIEW=0 \
  VIEW_MODE=none LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 \
  SPARSE_MIDDLE=0 LR=1e-3 NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 \
  TIME_BUDGET_SEC=600
```
One change vs. exp10: AdamW beta2 0.999 → 0.95. Removed vs. original
notebook: masked view, view term, laplacian, swd/bwd/c_swd/t_swd, sparse
middle; lr 2e-3 → 1e-3; guard margin 0.995 → 0.98; beta2 0.999 → 0.95.
Kept: 7.48M model, z 32×32, bs 64, full data, RoPE jitter, LR guard,
GRAD_CLIP=0.

Result at the 10-min mark (12,501 steps, 7.6 epochs, 39–42 ms/step,
~1,550–1,620 img/s; 51 ms on the 500-step intervals containing an epoch
boundary/probe):

| metric | exp12 (beta2 0.95) | exp10 (beta2 0.999) |
|---|---|---|
| FINAL train clean | **0.00640** | 0.00718 |
| FINAL test clean | **0.00635** | 0.00715 |
| step clean (last interval) | 0.00658 | 0.00729 |
| steps in 600 s | 12,501 | 12,818 |
| lr after E4 / E5 / E6 / E7 probe | 1e-3 / 1e-3 / 1e-3 / 1e-3 | 5e-4 / 5e-4 / 5e-4 / 5e-4 |

Per-epoch test clean, exp12 vs exp10: E1 0.02410/0.02228, E2 0.01773/0.01612,
E3 0.01315/0.01111, E4 0.01011/0.00970, E5 0.00828/0.00834, E6
**0.00746/0.01165**, E7 0.00690/0.00743, E8 0.00635/0.00715. Step clean at
500/1000/1500: 0.0537 / 0.0322 / 0.0274 (exp10: 0.0538 / 0.0319 / 0.0271).

LR probes (1× ratio): E4 0.9771, E5 0.9938, E6 0.9883, E7 0.9836 — all
< 1, so the guard kept lr 1e-3 the whole run (exp10: E4 1× 1.0678 → halved).
No loss spike anywhere: step-clean 500-step averages decrease monotonically
from step 8,200 (0.00878) to 12,480 (0.00658).

Implications:
- **New best: 0.00635, 11% better than exp10.** beta2 0.95 does exactly
  what it was tried for: lr 1e-3 stays stable through E7 (all probe ratios
  below 1) and the E6 spike that exp09/10/11 all reproduced is gone. The
  gain per epoch late in the run is +7.5–10% (E6–E8) vs exp10's +4–5% at
  5e-4.
- The cost is early progress: E1–E3 are 8–18% *behind* exp10 at the same
  step (the noisier second-moment estimate makes early updates less
  efficient), and exp12 only overtakes at E5. So beta2 0.95 is a
  "stability late" trade, not a per-step improvement, and it wins because
  it lets the run keep 2x the lr for the last 4 epochs.
- Wall-clock: 12,501 steps vs 12,818 in exp10 (~2.5% fewer); the log shows
  the interval ms/step at 39–42 vs 39, i.e. mild host contention, not a
  beta2 cost (fused AdamW does the same work regardless of beta values).
- Since lr 1e-3 is now stable with margin (E4 1× 0.977), the obvious next
  one-change step is to push lr up: exp13 = exp12 + `LR=2e-3` (the original
  notebook lr, which killed exp02 at beta2 0.999 / dense) — if beta2 0.95
  holds it, the late-run gain per epoch should rise again; if the guard
  halves it back to 1e-3 mid-run we lose little. Alternative: beta2 0.9 to
  see whether the early-epoch penalty shrinks or grows. Baseline for the
  loop is now **exp12 (test clean 0.00635)**; BETA2=0.95 joins NUM_WORKERS=4
  and GUARD_GAIN=0.98 as default knobs. Still 32x above 0.0002.

## 2026-09-18 — iteration 11: exp13 LR 2e-3 with beta2 0.95

No code change this iteration (all knobs already existed).

### exp13 — exp12 + LR=2e-3 (dense, pure recon, 4 workers, guard 0.98, beta2 0.95)

Command:
```
./run_exp.sh exp13_dense_lr2e-3_workers4_guard098_beta2_095 MASKED_VIEW=0 \
  VIEW_MODE=none LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 \
  SPARSE_MIDDLE=0 LR=2e-3 NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 \
  TIME_BUDGET_SEC=600
```
One change vs. exp12: lr 1e-3 → 2e-3 (this is the original notebook lr,
which diverged in exp02 at beta2 0.999). Removed vs. original notebook:
masked view, view term, laplacian, swd/bwd/c_swd/t_swd, sparse middle;
guard margin 0.995 → 0.98; beta2 0.999 → 0.95. Restored vs. exp12: the
notebook's lr 2e-3. Kept: 7.48M model, z 32×32, bs 64, full data, RoPE
jitter, LR guard, GRAD_CLIP=0.

Result at the 10-min mark (12,312 steps, 7.5 epochs, 39–44 ms/step on plain
intervals, 48–56 ms on intervals containing an epoch boundary/probe;
~1,600 img/s peak):

| metric | exp13 (lr 2e-3) | exp12 (lr 1e-3) |
|---|---|---|
| FINAL train clean | **0.00573** | 0.00640 |
| FINAL test clean | **0.00568** | 0.00635 |
| step clean (last interval) | 0.00583 | 0.00658 |
| steps in 600 s | 12,312 | 12,501 |
| lr after E4 / E5 / E6 / E7 probe | 2e-3 all four | 1e-3 all four |

Per-epoch test clean, exp13 vs exp12: E1 0.02208/0.02410, E2
**0.01327**/0.01773, E3 0.00975/0.01315, E4 0.00834/0.01011, E5
0.00702/0.00828, E6 0.00642/0.00746, E7 0.00599/0.00690, E8 0.00568/0.00635.
Step clean at 500/1000/1500: 0.0517 / 0.0307 / 0.0255 (exp12: 0.0537 /
0.0322 / 0.0274).

LR probes (1× ratio): E4 0.9859, E5 0.9928, E6 0.9872, E7 0.9849 — all
< 1 with margin comparable to exp12's (0.977–0.994), so the guard kept
2e-3 the whole run. No loss spike: step-clean 500-step averages are
monotone non-increasing except one flat interval (8,200 → 8,700: 0.00724 →
0.00725) and one +0.00002 blip at 11,980.

Implications:
- **New best: 0.00568, 10.5% better than exp12** and 21% better than exp10.
  With beta2 0.95, doubling lr to the notebook's 2e-3 is stable and
  ahead at *every* epoch — the lead is largest early (E2 −25%, E3 −26%)
  and shrinks late (E7 −13%, E8 −10.5%), i.e. the higher lr mostly buys
  faster early descent, and the late-run per-epoch gain is now
  +5–7%/epoch vs exp12's +7.5–10%. That is the usual signature of lr
  being near the edge of useful: the run is now limited by noise floor at
  this lr, not by step count.
- Throughput was ~1.5% lower than exp12 (12,312 vs 12,501 steps); the
  plain-interval ms/step (39–44) matches exp12, so this is host jitter
  again, not an lr cost.
- The probe margins did not shrink vs exp12 (E4 0.986 vs 0.977), so there
  is probably headroom for one more doubling, but the shrinking late-epoch
  gain argues the better next move is a decay schedule rather than more
  peak lr: exp14 = exp13 + `LR_SCHED=cosine` (knob exists; check
  `LR_MIN_FRAC`) so the final ~2 epochs anneal toward ~2e-4 and harvest
  the noise-floor gap. Alternative single-change candidates: `LR=4e-3`
  (tests the headroom directly), or `BETA2=0.9`. Baseline for the loop is
  now **exp13 (test clean 0.00568)**; default knobs are NUM_WORKERS=4,
  GUARD_GAIN=0.98, BETA2=0.95, LR=2e-3. Still 28x above 0.0002.

## 2026-09-18 — iteration 12: exp14 late cosine anneal (DECAY_START_FRAC)

**Infra (committed, no behaviour change when unset):** `train.py` reads
`DECAY_START_FRAC` (default 0): with `LR_SCHED=cosine`, lr stays at `LR`
(after warm-up) until that fraction of `TIME_BUDGET_SEC`, then cosine-decays
to `LR*LR_MIN_FRAC` at the budget. `WARMUP_STEPS=1` disables warm-up.
`run_exp.sh` header lists the knob.

### exp14 — exp13 + cosine anneal over the last 25% of the budget (2e-3 → 2e-4)

Command:
```
./run_exp.sh exp14_dense_lr2e-3_beta2_095_decay75 MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=2e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 LR_SCHED=cosine WARMUP_STEPS=1 \
  DECAY_START_FRAC=0.75 LR_MIN_FRAC=0.1 TIME_BUDGET_SEC=600
```
One change vs. exp13: constant lr 2e-3 → constant 2e-3 for 450 s, then cosine
to 2e-4 at 600 s (no warm-up). Removed vs. original notebook: masked view,
view term, laplacian, swd/bwd/c_swd/t_swd, sparse middle; guard margin
0.995 → 0.98; beta2 0.999 → 0.95; late lr anneal added. Kept: 7.48M model,
z 32×32, bs 64, full data, RoPE jitter, lr 2e-3, LR guard, GRAD_CLIP=0.

Result at the 10-min mark (12,706 steps, 7.75 epochs, 39–43 ms/step plain
intervals, 51 ms on probe intervals):

| metric | exp14 (anneal last 25%) | exp13 (const 2e-3) |
|---|---|---|
| FINAL train clean | 0.00574 | **0.00573** |
| FINAL test clean | 0.00569 | **0.00568** |
| step clean (last interval) | 0.00581 | 0.00583 |
| steps in 600 s | 12,706 | 12,312 |
| lr at E6 / E7 probe | 1.95e-3 / 8.0e-4 | 2e-3 / 2e-3 |

Per-epoch test clean, exp14 vs exp13: E1–E5 **identical to 4 digits**
(0.02208, 0.01327, 0.00975, 0.00834, 0.00702; deterministic, lr identical
until 450 s), E6 0.00645/0.00642, E7 **0.00585**/0.00599, E8 0.00569/0.00568.
Step-clean 500-step averages from 10,340: exp14 0.00644 / 0.00619 / 0.00605 /
0.00591 / 0.00588 / 0.00579 / 0.00581(226 steps); exp13 0.00652 / 0.00627 /
0.00623 / 0.00604 / 0.00606 / 0.00583(332 steps). Probes all kept scale 1.

Implications:
- **The anneal is a wash: 0.00569 vs 0.00568.** It buys −2.3% at E7 (lr
  ~1.2e-3 → 8e-4 over that epoch, 0.00585 vs 0.00599) but the gain
  evaporates in E8 as lr falls below ~6e-4: step clean flattens at
  0.00579–0.00581 for the last ~700 steps and E8 gains only +2.7% vs
  exp13's +5.2%. exp14 also had 3% more steps (host jitter), so per step
  it is marginally *worse* than constant lr at the end.
- This refutes the exp13 hypothesis that the run is noise-floor-limited at
  lr 2e-3. Lowering lr does not harvest a gap; the loss simply moves at a
  rate proportional to lr. Together with exp05 (full-budget cosine, −31%)
  the conclusion is: **for this problem within 10 min, any lr decay loses;
  keep lr as high as stability allows for the whole budget.**
- Therefore the remaining lr lever is upward: exp15 = exp13 + `LR=4e-3`
  (tests the headroom the probes suggest; E4–E7 1× ratios 0.985–0.993 in
  exp13). If 4e-3 diverges or the guard halves it, the lr axis is exhausted
  and the next levers are model cost (fewer blocks → more steps) or
  `BETA2=0.9`. Baseline for the loop stays **exp13 (test clean 0.00568)**;
  `LR_SCHED` stays `none` by default. Still 28x above 0.0002.

## 2026-09-18 — iteration 13: exp15 LR 4e-3 (lr headroom test)

No code change this iteration (all knobs already existed).

### exp15 — exp13 + LR=4e-3 (dense, pure recon, 4 workers, guard 0.98, beta2 0.95)

Command:
```
./run_exp.sh exp15_dense_lr4e-3_workers4_guard098_beta2_095 MASKED_VIEW=0 \
  VIEW_MODE=none LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 \
  SPARSE_MIDDLE=0 LR=4e-3 NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 \
  TIME_BUDGET_SEC=600
```
One change vs. exp13: lr 2e-3 → 4e-3. Removed vs. original notebook: masked
view, view term, laplacian, swd/bwd/c_swd/t_swd, sparse middle; guard margin
0.995 → 0.98; beta2 0.999 → 0.95; lr 2e-3 → 4e-3. Kept: 7.48M model, z 32×32,
bs 64, full data, RoPE jitter, LR guard, GRAD_CLIP=0.

Result at the 10-min mark (12,768 steps, 7.8 epochs, 39 ms/step, ~1,630 img/s):

| metric | exp15 (lr 4e-3) | exp13 (lr 2e-3) |
|---|---|---|
| FINAL train clean | 0.07171 | **0.00573** |
| FINAL test clean | 0.06840 | **0.00568** |
| step clean at 500 / 1000 / 1500 | 0.0682 / 0.0700 / 0.0702 | 0.0517 / 0.0307 / 0.0255 |
| lr after E4–E7 probes | 4e-3 all four | 2e-3 all four |

Per-epoch test clean: E1 0.06852, E2 0.06819, E3 0.06820, E4 0.06779, E5
0.06814, E6 0.06859, E7 0.06835, E8 0.06840 (best 0.06779 @E4). LR probes
(1× ratio): 0.9986 / 0.9992 / 0.9992 / 0.9992, every scale 0.999–1.000,
i.e. the loss is insensitive to lr — the dead-model signature of exp02.

Implications:
- **lr 4e-3 kills the model in the first 500 steps** (step clean 0.068 =
  mean-colour predictor, flat for the remaining 12k steps), exactly as lr
  2e-3 did at beta2 0.999 in exp02. beta2 0.95 bought one doubling of
  stable lr, not two. The guard cannot rescue a collapsed run because the
  probe measures sensitivity, and a dead model is insensitive to lr at every
  scale (all ratios ≈ 1.0), so it keeps scale 1.
- The collapse happens before step 500, so it is the same no-warm-up
  early-blow-up that exp03 diagnosed; exp05 showed a 300-step warm-up costs
  ~0.004 at E1 at lr 1e-3, but at 4e-3 it might be the difference between
  dead and alive. That is a possible follow-up (`LR=4e-3 LR_SCHED=cosine
  WARMUP_STEPS=300 DECAY_START_FRAC=1 LR_MIN_FRAC=1`, i.e. warm-up only),
  but the shrinking late gains at 2e-3 (exp13) mean the upside is bounded at
  ~10–15%; not enough to justify spending an iteration before the model-cost
  lever.
- **The constant-lr axis is exhausted: 2e-3 is the ceiling with beta2 0.95.**
  Remaining levers, in order: (1) model cost — `num_blocks` 8 → 4 or 6 to get
  ~1.5–2x the steps in 600 s (needs a `NUM_BLOCKS` env knob in the
  hyperparameter cell; note `sparse_middle` asserts shallow+final < blocks),
  (2) `BETA2=0.9`, (3) warm-up-only at 4e-3 as above. Next iteration:
  exp16 = exp13 + `NUM_BLOCKS=4`. Baseline for the loop stays **exp13 (test
  clean 0.00568)**; default knobs NUM_WORKERS=4, GUARD_GAIN=0.98, BETA2=0.95,
  LR=2e-3. Still 28x above 0.0002.

## 2026-09-18 — iteration 14: exp16 NUM_BLOCKS=4 (model-cost lever)

**Infra (committed, no behaviour change when unset):** `train.py` reads
`NUM_BLOCKS` (default 8, notebook) for the transformer depth. With
`SPARSE_MIDDLE=1` it must exceed shallow+final (2). `run_exp.sh` header lists
the knob.

### exp16 — exp13 + NUM_BLOCKS=4 (dense, pure recon, 4 workers, guard 0.98, beta2 0.95, lr 2e-3)

Command:
```
./run_exp.sh exp16_dense_lr2e-3_beta2_095_blocks4 MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=2e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 NUM_BLOCKS=4 TIME_BUDGET_SEC=600
```
One change vs. exp13: 8 blocks → 4 blocks (7.48M → 3.92M params). Removed vs.
original notebook: masked view, view term, laplacian, swd/bwd/c_swd/t_swd,
sparse middle, half the transformer depth; guard margin 0.995 → 0.98; beta2
0.999 → 0.95. Kept: z 32×32, bs 64, full data, RoPE jitter, lr 2e-3, LR
guard, GRAD_CLIP=0.

Result at the 10-min mark (20,979 steps, 12.8 epochs, 23–29 ms/step plain
intervals, ~2,200–2,800 img/s, 0.61–0.77 min/epoch):

| metric | exp16 (4 blocks) | exp13 (8 blocks) |
|---|---|---|
| FINAL train clean | **0.00530** | 0.00573 |
| FINAL test clean | **0.00533** | 0.00568 |
| steps in 600 s | 20,979 | 12,312 |
| ms/step (plain) | 23–29 | 39–43 |
| step clean at 500 / 1000 / 1500 | 0.0542 / 0.0344 / 0.0280 | 0.0517 / 0.0307 / 0.0255 |
| lr after probes | 2e-3 all nine | 2e-3 all four |

Per-epoch test clean, exp16: E1 0.02331, E2 0.01742, E3 0.01266, E4 0.01040,
E5 0.00908, E6 0.00822, E7 0.00760, E8 0.00699, E9 0.00650, E10 0.00615,
E11 0.00572, E12 0.00548, E13 0.00533. Per epoch it is worse than exp13
(E1–E5: 0.02208, 0.01327, 0.00975, 0.00834, 0.00702) but per wall clock it
is ahead from ~4 min: exp16 at 3.7 min (E5) 0.00908 vs exp13 at 3.9 min (E3)
0.00975; at 5.7 min (E8) 0.00699 vs 6.4 min (E5) 0.00702; at 7.9 min (E11)
0.00572 vs exp13's 10-min 0.00568. LR probes: 1× ratios 0.975–0.996, all
scales kept at 1, no instability (a 4-block model at 2e-3 is as stable as 8
blocks).

Implications:
- **Halving depth is a −6% win (0.00533 vs 0.00568), new best.** 1.7x the
  steps buys more than the capacity loss costs within 10 min; per-step the
  small model is ~15% worse early but the gap closes to ~0 by 20k steps
  (exp13 step clean 0.00583 at 12.3k vs exp16 0.00542 at 21k).
- Step time did not halve (39 → 26 ms; 1.5x not 2x), so the loader / eval /
  non-transformer parts (patch embed, decoder, dataloader at 4 workers) are
  now a larger share. 500-step intervals with img/s 2,800 mean the GPU is
  doing 2.2k–2.8k img/s; exp08 showed the loader saturating around there.
- Late gains per epoch are still small (+2.7%/epoch at E13), so the run is
  again step-rate-limited, not converged. Next lever candidates, in order:
  (1) `NUM_BLOCKS=2` (tests whether the trend continues, or whether 4 is the
  knee), (2) `NUM_BLOCKS=4` + `BATCH_SIZE=128` (fewer, bigger steps; with a
  faster model the loader overhead per step matters more), (3) `NUM_WORKERS=8`
  to check whether the loader is the bottleneck at 2.8k img/s. Next
  iteration: exp17 = exp16 + `NUM_BLOCKS=2`. Baseline for the loop is now
  **exp16 (test clean 0.00533)**; default knobs NUM_WORKERS=4,
  GUARD_GAIN=0.98, BETA2=0.95, LR=2e-3, NUM_BLOCKS=4. Still 27x above 0.0002.

## 2026-09-18 — iteration 15: exp17 NUM_BLOCKS=2 (depth knee test)

**Infra (committed, no behaviour change at the notebook default):**
`SparseMiddleTransformer.__init__` used to require shallow+final < num_blocks
(≥1 middle block), which crashed `NUM_BLOCKS=2` at construction
("Need at least one middle block."). Relaxed to `shallow+final <= num_blocks`:
with zero middle blocks the middle loop is empty, `deep = shallow`, and the
fusion is identity at init, so the dense path is well-defined (sparse mode
with 0 middle blocks would be pointless but is not asserted against). Crash
log kept as `artifacts/logs/exp17_crash_blocks2_assert.log`.

### exp17 — exp16 + NUM_BLOCKS=2 (dense, pure recon, 4 workers, guard 0.98, beta2 0.95, lr 2e-3)

Command:
```
./run_exp.sh exp17_dense_lr2e-3_beta2_095_blocks2 MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=2e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 NUM_BLOCKS=2 TIME_BUDGET_SEC=600
```
One change vs. exp16: 4 blocks → 2 blocks (3.92M → 2.14M params; 1 shallow +
0 middle + 1 final per core). Removed vs. original notebook: masked view, view
term, laplacian, swd/bwd/c_swd/t_swd, sparse middle, 3/4 of the transformer
depth; guard margin 0.995 → 0.98; beta2 0.999 → 0.95. Kept: z 32×32, bs 64,
full data, RoPE jitter, lr 2e-3, LR guard, GRAD_CLIP=0.

Result at the 10-min mark (30,122 steps, 18.4 epochs, 13–18 ms/step plain
intervals (mode 17), ~3,900 img/s peak, 0.41–0.63 min/epoch):

| metric | exp17 (2 blocks) | exp16 (4 blocks) | exp13 (8 blocks) |
|---|---|---|---|
| FINAL train clean | 0.00809 | **0.00530** | 0.00573 |
| FINAL test clean | 0.00820 (best 0.00783 @E18) | **0.00533** | 0.00568 |
| steps in 600 s | 30,122 | 20,979 | 12,312 |
| ms/step (plain) | 13–18 | 23–29 | 39–43 |
| step clean at 500 / 1000 / 1500 | 0.0550 / 0.0358 / 0.0318 | 0.0542 / 0.0344 / 0.0280 | 0.0517 / 0.0307 / 0.0255 |
| lr after probes | 2e-3 all 30 | 2e-3 all nine | 2e-3 all four |

Per-epoch test clean, exp17: E1 0.02881, E2 0.02518, E3 0.02191, E4 0.01982,
E5 0.01599, E6 0.01428, E7 0.01285, E8 0.01208, E9 0.01102, E10 0.01033,
E11 0.00994, E12 0.00954, E13 0.00917, E14 0.00878, E15 0.00856, E16 0.00826,
E17 0.00813, E18 0.00783, E19 (partial, 0.16 min) 0.00820. LR probes: 1×
ratios 0.967–0.997, scale kept at 1 throughout; no instability.

Implications:
- **2 blocks is past the knee: +54% worse than 4 blocks (0.00820 vs 0.00533).**
  1.44x the steps does not compensate; at equal wall clock exp17 is behind
  exp16 at every point (exp17 E13 @ ~6 min 0.00917 vs exp16 E8 @ 5.7 min
  0.00699). Per epoch it is ~2x worse from E3 on (0.0219 vs 0.0127) and the
  gap does not close with steps (exp17 at 21k steps ≈ E13 0.00917 vs exp16
  at 21k 0.00533). The loss curve flattens (+2–4%/epoch from E13), i.e. the
  2-block model is capacity-limited, not step-limited.
- Step time fell 26 → 17 ms (1.5x) for a 2x depth cut, and img/s peaked at
  3,900 with 500-step intervals down to 13 ms. The non-transformer floor
  (patch embed, decoder conv, loader, optimizer) is now ~10 ms/step, so
  further depth cuts cannot buy many steps anyway.
- **Depth: 4 is the wall-clock optimum among {2, 4, 8}; 6 is untested but
  bracketed** (8 → 4 gave −6%, 4 → 2 gave +54%, so 6 could be at most a small
  gain; low priority). Remaining levers, in order: (1) `NUM_BLOCKS=4` +
  `BATCH_SIZE=128` (per-step overhead now dominates: 26 ms/step at bs 64 with
  a ~10 ms floor, so doubling the batch should cost far less than 2x per
  step while halving optimizer/loader overhead per image; keep lr 2e-3 first,
  then try 4e-3 with the bigger batch since larger batches tolerate higher
  lr), (2) `EMB_DIM` width reduction on 4 blocks (same argument as depth, but
  width cuts attention/MLP cost quadratically), (3) `NUM_WORKERS=8` if the
  bs-128 run shows the loader capping img/s. Next iteration: exp18 = exp16 +
  `BATCH_SIZE=128`. Baseline for the loop stays **exp16 (test clean
  0.00533)**; default knobs NUM_WORKERS=4, GUARD_GAIN=0.98, BETA2=0.95,
  LR=2e-3, NUM_BLOCKS=4. Still 27x above 0.0002.

## 2026-09-18 — iteration 16: exp18 BATCH_SIZE=128 (fewer, bigger steps)

No code change; `BATCH_SIZE` knob already existed.

### exp18 — exp16 + BATCH_SIZE=128 (dense, pure recon, 4 blocks, 4 workers, guard 0.98, beta2 0.95, lr 2e-3)

Command:
```
./run_exp.sh exp18_dense_lr2e-3_beta2_095_blocks4_bs128 MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=2e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 NUM_BLOCKS=4 BATCH_SIZE=128 TIME_BUDGET_SEC=600
```
One change vs. exp16: batch 64 → 128 (3.92M params, same model). Removed vs.
original notebook: masked view, view term, laplacian, swd/bwd/c_swd/t_swd,
sparse middle, half the transformer depth; guard margin 0.995 → 0.98; beta2
0.999 → 0.95; bs 64 → 128. Kept: z 32×32, full data, RoPE jitter, lr 2e-3,
LR guard, GRAD_CLIP=0.

Result at the 10-min mark (10,661 steps, 13.0 epochs, 45–52 ms/step plain
intervals (mode 48), 2,600–2,800 img/s, 0.63–0.68 min/epoch):

| metric | exp18 (bs 128) | exp16 (bs 64) |
|---|---|---|
| FINAL train clean | 0.00746 | **0.00530** |
| FINAL test clean | 0.00732 | **0.00533** |
| steps in 600 s | 10,661 | 20,979 |
| epochs in 600 s | 13.0 | 12.8 |
| ms/step (plain) | 45–52 | 23–29 |
| img/s | 2,600–2,800 | 2,200–2,800 |
| step clean at 500 / 1640 / 4920 | 0.0530 / 0.0240 / 0.0110 | 0.0542 / — / — |
| lr after probes | 2e-3 all seven | 2e-3 all nine |

Per-epoch test clean, exp18: E1 0.03002, E2 0.02200, E3 0.01888, E4 0.01544,
E5 0.01218, E6 0.01040, E7 0.00987, E8 0.00923, E9 0.00859, E10 0.00821,
E11 0.00804, E12 0.00752, E13 0.00739, E14 (partial) 0.00732. Versus exp16
per epoch (E1 0.02331, E5 0.00908, E10 0.00615, E13 0.00533) it is 25–37%
worse at every epoch. LR probes: 1× ratios 0.976–0.998, scale kept at 1; no
instability.

Implications:
- **Doubling the batch is a +37% loss (0.00732 vs 0.00533). Rejected.**
  Epoch time is unchanged (0.65 vs 0.61–0.77 min): step time went 26 → 48 ms,
  i.e. almost exactly 2x for 2x the images. So the "~10 ms fixed floor per
  step" hypothesis from iteration 15 was wrong for the 4-block model: cost is
  ~0.36 ms/image + ~5 ms fixed, the GPU is compute-bound at ~2,700 img/s at
  this depth, and bigger batches buy nothing in images/s. With the same
  images seen, half the optimizer updates → worse per epoch, exactly as with
  fewer steps in general. The loader is not the cap either (exp17 reached
  3,900 img/s with the same 4 workers).
- Consequence: within 10 min the objective is *updates per second at fixed
  compute*, and compute per image is fixed by the model. So the remaining
  levers are (a) **smaller batch** (`BATCH_SIZE=32` → ~16 ms/step → ~36k
  steps; the per-step-vs-per-image trade goes the other way, and the LR
  guard/probes will tell if 2e-3 is still stable at bs 32), (b) cheaper
  per-image compute at the same depth: `EMB_DIM` reduction (attention/MLP
  cost ~quadratic in width) or fewer latent tokens (`N_LATENT_TOKENS`),
  (c) lr 4e-3 with warm-up (bounded upside). Not worth trying: larger
  batches, more loader workers.
- Next iteration: exp19 = exp16 + `BATCH_SIZE=32` (one change; if it wins,
  bs 16 next; if 2e-3 destabilises at bs 32, the guard should step it down
  and that is itself the answer). Baseline for the loop stays **exp16 (test
  clean 0.00533)**; default knobs NUM_WORKERS=4, GUARD_GAIN=0.98, BETA2=0.95,
  LR=2e-3, NUM_BLOCKS=4, BATCH_SIZE=64. Still 27x above 0.0002.

## 2026-09-18 — iteration 17: exp19 BATCH_SIZE=32 (more, smaller steps)

No code change; `BATCH_SIZE` knob already existed.

### exp19 — exp16 + BATCH_SIZE=32 (dense, pure recon, 4 blocks, 4 workers, guard 0.98, beta2 0.95, lr 2e-3)

Command:
```
./run_exp.sh exp19_dense_lr2e-3_beta2_095_blocks4_bs32 MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=2e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 NUM_BLOCKS=4 BATCH_SIZE=32 TIME_BUDGET_SEC=600
```
One change vs. exp16: batch 64 → 32 (3.92M params, same model). Removed vs.
original notebook: masked view, view term, laplacian, swd/bwd/c_swd/t_swd,
sparse middle, half the transformer depth; guard margin 0.995 → 0.98; beta2
0.999 → 0.95; bs 64 → 32. Kept: z 32×32, full data, RoPE jitter, lr 2e-3,
LR guard, GRAD_CLIP=0.

Result at the 10-min mark (32,370 steps, 9.9 epochs, 12–23 ms/step plain
intervals (mode 13), 1,400–2,700 img/s, 0.69–1.27 min/epoch):

| metric | exp19 (bs 32) | exp16 (bs 64) | exp18 (bs 128) |
|---|---|---|---|
| FINAL train clean | 0.01839 | **0.00530** | 0.00746 |
| FINAL test clean | 0.01839 (best 0.01809 @E9) | **0.00533** | 0.00732 |
| steps in 600 s | 32,370 | 20,979 | 10,661 |
| epochs in 600 s | 9.9 | 12.8 | 13.0 |
| ms/step (plain) | 12–23 | 23–29 | 45–52 |
| img/s | 1,400–2,700 | 2,200–2,800 | 2,600–2,800 |
| step clean at 500 / 1000 / 1500 | 0.0689 / 0.0671 / 0.0672 | 0.0542 / 0.0344 / 0.0280 | 0.0530 / — / — |
| lr after probes | 2e-3 all eight | 2e-3 all nine | 2e-3 all seven |

Per-epoch test clean, exp19: E1 0.05597, E2 0.05724, E3 0.04096, E4 0.03331,
E5 0.02828, E6 0.02592, E7 0.02143, E8 0.01999, E9 0.01809, E10 0.01839.
Versus exp16 (E1 0.02331, E5 0.00908, E10 0.00615) it is 2.4–3.1x worse at
every epoch. LR probes: 1× ratios 0.951–0.998, scale kept at 1 (the guard
never fired, so lr 2e-3 was "stable" by its criterion, yet see below).

Implications:
- **Halving the batch is a 3.45x loss (0.01839 vs 0.00533). Rejected hard.**
  Two independent failures:
  1. *Optimisation*: the step-clean curve shows the exp15 (lr 4e-3, bs 64)
     "dead model" signature — flat at 0.067 from step 500 to 2,500, only
     starting to move at step ~3,000. lr/batch doubled, so the effective
     noise scale is the same as the lr 4e-3 run; here it recovered (the
     guard's 1× ratio dipped to 0.951 at E4 but never crossed 0.98) but
     lost ~2 epochs of progress it never made back. So lr 2e-3 at bs 64 is
     sitting right at the stability edge and cannot be pushed further in
     either lr or 1/batch.
  2. *Throughput*: ms/step fell 26 → 13–16 (mode 13), i.e. only ~1.7x for a
     2x smaller batch, and step time was noisy (21–23 ms intervals during
     E2–E4, likely loader/launch bound), so img/s dropped to 1,400–2,700 and
     the run saw fewer epochs (9.9 vs 12.8). Per-step floor is ~5–8 ms at
     this model size.
- Batch axis is now bracketed {32, 64, 128} with 64 the clear optimum;
  step-count-per-second beyond ~35 steps/s buys nothing because both the
  data seen and the stable lr shrink. Do not revisit batch size.
- Remaining levers, in order: (a) cheaper per-image compute at 4 blocks and
  bs 64 — `N_LATENT_TOKENS` 32 → 16 (attention cost scales with tokens; the
  latent bottleneck may or may not be what limits clean recon) or `EMB_DIM`
  32 → 24/16 (width); (b) the untested NUM_BLOCKS=6 (bounded gain);
  (c) lr warm-up + 3e-3 at bs 64 (exp15 at 4e-3 died without warm-up; a
  short warm-up might rescue 3e-3 — but the bs-32 result says the edge is
  close, low priority). Next iteration: exp20 = exp16 + `N_LATENT_TOKENS=16`
  (one change). Baseline for the loop stays **exp16 (test clean 0.00533)**;
  default knobs NUM_WORKERS=4, GUARD_GAIN=0.98, BETA2=0.95, LR=2e-3,
  NUM_BLOCKS=4, BATCH_SIZE=64. Still 27x above 0.0002.

## 2026-09-18 — iteration 18: exp20 N_LATENT_TOKENS=16 (smaller latent bottleneck)

No code change; `N_LATENT_TOKENS` knob already existed.

### exp20 — exp16 + N_LATENT_TOKENS=16 (dense, pure recon, 4 blocks, bs 64, 4 workers, guard 0.98, beta2 0.95, lr 2e-3)

Command:
```
./run_exp.sh exp20_dense_lr2e-3_beta2_095_blocks4_lat16 MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=2e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 NUM_BLOCKS=4 N_LATENT_TOKENS=16 TIME_BUDGET_SEC=600
```
One change vs. exp16: z tokens 32 → 16, so z total dim 1,024 → 512 and the
transformer sequence 96 → 80 tokens (64 patches + 16 latents). Param count
unchanged at 3,917,360 (latent tokens add no weights). Removed vs. original
notebook: masked view, view term, laplacian, swd/bwd/c_swd/t_swd, sparse
middle, half the transformer depth, half the latent tokens; guard margin
0.995 → 0.98; beta2 0.999 → 0.95. Kept: full data, RoPE jitter, bs 64,
lr 2e-3, LR guard, GRAD_CLIP=0.

Result at the 10-min mark (20,173 steps, 12.3 epochs, 21–32 ms/step plain
intervals, 2,000–3,100 img/s, 0.60–0.81 min/epoch):

| metric | exp20 (z 16 tok) | exp16 (z 32 tok) |
|---|---|---|
| FINAL train clean | 0.00888 | **0.00530** |
| FINAL test clean | 0.00896 | **0.00533** |
| steps in 600 s | 20,173 | 20,979 |
| epochs in 600 s | 12.3 | 12.8 |
| ms/step (plain) | 21–32 | 23–29 |
| step clean at 500 / 1000 / 1640 | 0.0568 / 0.0368 / 0.0295 | 0.0542 / 0.0344 / — |
| lr after probes | 2e-3 all nine | 2e-3 all nine |

Per-epoch test clean, exp20: E1 0.02835, E2 0.02286, E3 0.01779, E4 0.01437,
E5 0.01249, E6 0.01164, E7 0.01104, E8 0.01031, E9 0.01011, E10 0.00989,
E11 0.00927, E12 0.00897, E13 (partial) 0.00896. Versus exp16 (E1 0.02331,
E3 0.01266, E5 0.00908, E8 0.00699, E10 0.00615, E13 0.00533) it is 22–68%
worse at every epoch and the gap widens with training (E1 +22%, E5 +38%,
E10 +61%, E13 +68%); per-epoch gain has fallen to +0–3% by E9–E13, i.e. it is
converging to a higher floor. LR probes: 1× ratios 0.983–1.017, scale kept
at 1; no instability.

Implications:
- **Halving the latent tokens is a +68% loss (0.00896 vs 0.00533) for zero
  speed-up. Rejected.** Step time did not move (20.2k vs 21.0k steps; the
  16-token drop is a 17% sequence cut, and attention over ≤96 tokens at
  width 32 is a small share of per-step cost next to the MLP/patch/decoder
  work). So the latent token count is not a throughput lever at all.
- It *is* a capacity lever: the same model with a 512-dim z instead of
  1,024 flattens early and the curve is capacity-limited (like the 2-block
  exp17), not step-limited. That means the z bottleneck is one of the things
  currently holding clean recon at ~5e-3, which points the other way: more
  latent tokens are nearly free in wall clock and may lower the floor.
- Next iteration: exp21 = exp16 + `N_LATENT_TOKENS=64` (z total dim 2,048,
  sequence 128; expect ≤10% step-time cost). If it wins, go to 128 next;
  if it does not, width (`EMB_DIM`) is the remaining capacity/compute lever
  and NUM_BLOCKS=6 the last depth point. Baseline for the loop stays
  **exp16 (test clean 0.00533)**; default knobs NUM_WORKERS=4,
  GUARD_GAIN=0.98, BETA2=0.95, LR=2e-3, NUM_BLOCKS=4, BATCH_SIZE=64,
  N_LATENT_TOKENS=32. Still 27x above 0.0002.

## 2026-09-18 — iteration 19: exp21 N_LATENT_TOKENS=64 (larger latent bottleneck)

No code change; `N_LATENT_TOKENS` knob already existed.

### exp21 — exp16 + N_LATENT_TOKENS=64 (dense, pure recon, 4 blocks, bs 64, 4 workers, guard 0.98, beta2 0.95, lr 2e-3)

Command:
```
./run_exp.sh exp21_dense_lr2e-3_beta2_095_blocks4_lat64 MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=2e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 NUM_BLOCKS=4 N_LATENT_TOKENS=64 TIME_BUDGET_SEC=600
```
One change vs. exp16: z tokens 32 → 64, so z total dim 1,024 → 2,048 and the
transformer sequence 96 → 128 tokens. Params 3,923,504 (vs 3,917,360; the
+6k is the larger latent embedding table). Removed vs. original notebook:
masked view, view term, laplacian, swd/bwd/c_swd/t_swd, sparse middle, half
the transformer depth; guard margin 0.995 → 0.98; beta2 0.999 → 0.95; latent
tokens doubled. Kept: full data, RoPE jitter, bs 64, lr 2e-3, LR guard,
GRAD_CLIP=0.

Result at the 10-min mark (21,321 steps, 13.0 epochs, 24–26 ms/step plain
intervals, 2,500–2,650 img/s, 0.61–0.80 min/epoch):

| metric | exp21 (z 64 tok) | exp16 (z 32 tok) | exp20 (z 16 tok) |
|---|---|---|---|
| FINAL train clean | 0.00711 | **0.00530** | 0.00888 |
| FINAL test clean | 0.00698 | **0.00533** | 0.00896 |
| steps in 600 s | 21,321 | 20,979 | 20,173 |
| epochs in 600 s | 13.0 | 12.8 | 12.3 |
| ms/step (plain) | 24–26 | 23–29 | 21–32 |
| step clean at 500 / 1000 / 1640 | 0.0621 / 0.0384 / 0.0272 | 0.0542 / 0.0344 / — | 0.0568 / 0.0368 / 0.0295 |
| lr after probes | **1e-3 from E4 on** (guard fired once) | 2e-3 all nine | 2e-3 all nine |

Per-epoch test clean, exp21: E1 0.02617, E2 0.01697, E3 0.01335, E4 0.01143,
E5 0.00991, E6 0.00937, E7 0.00889, E8 0.00840, E9 0.00807, E10 0.00771,
E11 0.00750, E12 0.00719, E13 0.00699, E14 (partial) 0.00698. Versus exp16
(E1 0.02331, E3 0.01266, E5 0.00908, E8 0.00699, E10 0.00615, E13 0.00533)
it is +5–12% worse through E5 and +20–31% worse from E8 on.

LR guard: the first probe (step 6,560, end of E4) measured 1× ratio 1.0159
(0.75× 1.0004, 0.5× 0.9928) — i.e. continuing at lr 2e-3 made the EMA loss
*worse* — and halved lr to 1e-3. exp16's same probe read 1× 0.9753 and kept
2e-3. All later probes (8,200 … 21,320) kept 1e-3 with 1× ratios 0.991–1.006.

Implications:
- **Doubling the latent tokens is a +31% loss (0.00698 vs 0.00533) at the
  same throughput. Rejected.** Step time is flat (24–26 ms vs 23–29), so as
  in exp20 the token count is not a speed lever; the 128-token sequence
  costs nothing measurable at width 32.
- The loss has two parts. (1) Even before the guard fired, E1–E5 were 5–12%
  behind exp16, so the 2,048-dim z did not lower the early curve; the
  bottleneck at 32 tokens is not binding early. (2) At E4 the run hit an
  lr-2e-3 instability the 32-token model never showed (1× probe ratio 1.016
  vs 0.975), and the remaining 9 epochs at lr 1e-3 opened the gap to +31%.
  So more latent tokens make lr 2e-3 *less* stable (a longer sequence → more
  attention mass per patch token → larger effective updates), and the
  guard's half-step costs more than the extra capacity is worth here.
- Latent-token axis is now bracketed {16, 32, 64} with 32 the clear optimum
  (0.00896 / 0.00533 / 0.00698). Do not revisit N_LATENT_TOKENS in phase 1.
  exp20's inference that the z bottleneck was "one of the things holding
  clean recon at ~5e-3" is *not* supported by this run: relaxing it did not
  help even before the lr cut.
- Remaining levers, in order: (a) width `EMB_DIM` 32 → 48 or 64 (the one
  capacity lever untested; attention/MLP cost grows ~quadratically so expect
  a real step-time cost, but exp17/exp20 show the run is capacity-limited by
  ~E9, so trading steps for width may pay); (b) NUM_BLOCKS=6 (bounded);
  (c) a short lr warm-up so 2e-3 is stable from step 0 (exp15/exp19/exp21 all
  show the stability edge at 2e-3·bs64; warm-up is cheap and might allow
  3e-3 after it). Next iteration: exp22 = exp16 + `EMB_DIM=48` (one change).
  Baseline for the loop stays **exp16 (test clean 0.00533)**; default knobs
  NUM_WORKERS=4, GUARD_GAIN=0.98, BETA2=0.95, LR=2e-3, NUM_BLOCKS=4,
  BATCH_SIZE=64, N_LATENT_TOKENS=32, EMB_DIM=32. Still 27x above 0.0002.

## 2026-09-18 — iteration 20: exp22 EMB_DIM=48 (turned out to be the z channel width, not transformer width)

Code change (after the run, for the next iteration): `train.py` gains env
knobs `AE_EMB_DIM` (transformer width, notebook 192), `N_HEADS` (8) and
`HEAD_SIZE` (24, must be divisible by 4 for 2D RoPE); `run_exp.sh` header
lists them. The exp22 run itself used no code change.

### exp22 — exp16 + EMB_DIM=48 (dense, pure recon, 4 blocks, bs 64, 4 workers, guard 0.98, beta2 0.95, lr 2e-3)

Command:
```
./run_exp.sh exp22_dense_lr2e-3_beta2_095_blocks4_emb48 MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=2e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 NUM_BLOCKS=4 EMB_DIM=48 TIME_BUDGET_SEC=600
```
**Attribution correction.** `EMB_DIM` is *not* the transformer width. In
`train.py` the transformer runs at `ae_emb_dim = 192` (8 heads × 24) and
`emb_dim` is only the z channel width used by `emb_out_proj`
(192 → emb_dim) and the decoder's input projection. So exp22 changed the z
shape (32, 32) → (32, 48), z total dim 1,024 → 1,536, and params moved
3,917,360 → 3,925,552 (+8k). It is therefore a third point on the same
z-total-dim axis as exp20 (512) and exp21 (2,048), not a width test.
Removed vs. original notebook: masked view, view term, laplacian,
swd/bwd/c_swd/t_swd, sparse middle, half the depth; guard 0.995 → 0.98;
beta2 0.999 → 0.95; z channel 32 → 48. Kept: full data, RoPE jitter, bs 64,
lr 2e-3, LR guard, GRAD_CLIP=0, transformer width 192.

Result at the 10-min mark (21,432 steps, 13.1 epochs, 24–32 ms/step plain
intervals, 2,000–2,700 img/s, 0.58–0.74 min/epoch):

| metric | exp22 (z 32×48 = 1,536) | exp16 (z 32×32 = 1,024) | exp21 (z 64×32 = 2,048) | exp20 (z 16×32 = 512) |
|---|---|---|---|---|
| FINAL train clean | 0.00707 | **0.00530** | 0.00711 | 0.00888 |
| FINAL test clean | 0.00691 | **0.00533** | 0.00698 | 0.00896 |
| steps in 600 s | 21,432 | 20,979 | 21,321 | 20,173 |
| ms/step (plain) | 24–32 | 23–29 | 24–26 | 21–32 |
| step clean at 500 / 1000 / 1640 | 0.0578 / 0.0351 / 0.0271 | 0.0542 / 0.0344 / — | 0.0621 / 0.0384 / 0.0272 | 0.0568 / 0.0368 / 0.0295 |
| lr after probes | 2e-3 all ten | 2e-3 all nine | 1e-3 from E4 | 2e-3 all nine |

Per-epoch test clean, exp22: E1 0.02587, E2 0.01954, E3 0.01548, E4 0.01239,
E5 0.01078, E6 0.00975, E7 0.00904, E8 0.00865, E9 0.00848, E10 0.00811,
E11 0.00751, E12 0.00738, E13 0.00699, E14 (partial) 0.00691. Versus exp16
(E1 0.02331, E3 0.01266, E5 0.00908, E8 0.00699, E10 0.00615, E13 0.00533):
+11% at E1, +22% at E3, +19% at E5, +24% at E8, +32% at E10, +31% at E13.
LR probes: the E4 probe read 1× 1.0122 (0.75× 0.9988, 0.5× 0.9925) — the
same lr-2e-3 wobble exp21 hit — but no scale beat 1× by the 2% guard margin
so lr stayed 2e-3; all later 1× ratios 0.973–0.995. So unlike exp21 the gap
here is *not* an lr-cut artifact: at full lr throughout, a wider z is still
+30% worse.

Implications:
- **z channel width 48 is a +30% loss (0.00691 vs 0.00533) at equal
  throughput. Rejected.** Together with exp20/exp21 the z-total-dim axis is
  now {512: 0.00896, 1,024: 0.00533, 1,536: 0.00691, 2,048: 0.00698}: a sharp
  optimum at 1,024 with either shape. Enlarging z hurts regardless of whether
  it comes from more tokens or wider channels, and it is not a stability
  story (exp22 kept lr 2e-3). Best reading: the 192→32 projection with a
  1,024-dim z acts as a useful regulariser/conditioning for the decoder at
  this step budget; wider z slows the early curve (E1 already +11%) and never
  catches up. Do not revisit z size in phase 1.
- The true width lever (`ae_emb_dim` 192, and heads × head_size) was never
  tested; it is now exposed as `AE_EMB_DIM` / `N_HEADS` / `HEAD_SIZE`. The
  transformer at width 192 is where the per-step compute goes, so this is
  the first experiment that trades steps for capacity (or, going down,
  capacity for steps) for real.
- Next iteration: exp23 = exp16 + `AE_EMB_DIM=256` (must keep n_heads ×
  head_size consistent only through the QKV/out_proj linears, which are
  sized independently, so 256 is valid with 8 × 24 heads). If the step-time
  cost is ≤30% and the curve is ahead by E5, follow with 320; if it is
  behind, try `AE_EMB_DIM=128` for the speed direction. Then NUM_BLOCKS=6
  and lr warm-up. Baseline for the loop stays **exp16 (test clean
  0.00533)**; default knobs NUM_WORKERS=4, GUARD_GAIN=0.98, BETA2=0.95,
  LR=2e-3, NUM_BLOCKS=4, BATCH_SIZE=64, N_LATENT_TOKENS=32, EMB_DIM=32,
  AE_EMB_DIM=192. Still 27x above 0.0002.

## 2026-09-18 — iteration 21: exp23 AE_EMB_DIM=256 (true transformer width, 192 → 256)

No code change; `AE_EMB_DIM` knob added last iteration.

### exp23 — exp16 + AE_EMB_DIM=256 (dense, pure recon, 4 blocks, bs 64, 4 workers, guard 0.98, beta2 0.95, lr 2e-3)

Command:
```
./run_exp.sh exp23_dense_lr2e-3_beta2_095_blocks4_ae256 MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=2e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 NUM_BLOCKS=4 AE_EMB_DIM=256 TIME_BUDGET_SEC=600
```
One change vs. exp16: transformer width 192 → 256 (heads stay 8 × 24; z
stays 32 × 32 = 1,024). Params 6,336,944 (vs 3,917,360, +62%). Removed vs.
original notebook: masked view, view term, laplacian, swd/bwd/c_swd/t_swd,
sparse middle, half the depth; guard 0.995 → 0.98; beta2 0.999 → 0.95.
Changed: transformer width 192 → 256 (wider than notebook). Kept: full
data, RoPE jitter, bs 64, lr 2e-3, LR guard, GRAD_CLIP=0, z 1,024.

Result at the 10-min mark (17,642 steps, 10.8 epochs, 27–33 ms/step plain
intervals, 1,960–2,330 img/s, 0.79–0.87 min/epoch):

| metric | exp23 (width 256) | exp16 (width 192) |
|---|---|---|
| FINAL train clean | 0.01029 | **0.00530** |
| FINAL test clean | 0.01006 | **0.00533** |
| steps in 600 s | 17,642 | 20,979 |
| epochs in 600 s | 10.8 | 12.8 |
| ms/step (plain) | 27–33 | 23–29 |
| step clean at 500 / 1000 / 1640 | 0.0594 / 0.0396 / 0.0297 | 0.0542 / 0.0344 / — |
| lr after probes | 2e-3 all eight | 2e-3 all nine |

Per-epoch test clean, exp23: E1 0.02997, E2 0.02382, E3 0.02041, E4 0.01671,
E5 0.01397, E6 0.01221, E7 0.01140, E8 0.01075, E9 0.01034, E10 0.01012,
E11 0.01006. Versus exp16 (E1 0.02331, E3 0.01266, E5 0.00908, E8 0.00699,
E10 0.00615): +29% at E1, +61% at E3, +54% at E5, +54% at E8, +65% at E10.
LR probes were all clean (1× ratios 0.986–1.003, no scale beat 1× by the 2%
margin), so this is not a stability artifact.

Implications:
- **Width 256 is a +89% loss (0.01006 vs 0.00533). Rejected.** Both halves
  hurt: step time +15–20% (16% fewer steps) *and* the per-epoch curve is
  behind by E1 and never closes — at equal epoch count (E10) it is +65%
  worse. So at lr 2e-3 / bs 64 the wider transformer learns *slower per
  step*, not just per second. Consistent with exp17 (depth 8 → worse) and
  exp20–22 (bigger z → worse): every capacity increase from the exp16
  config has lost, and by a margin that grows with size. The 10-minute
  budget is an optimization-speed regime, not a capacity regime, and the
  wider model most likely needs a smaller lr per parameter (µP-style) that
  the fixed 2e-3 does not provide.
- The interesting direction is therefore the *speed* side: `AE_EMB_DIM=128`
  (≈2.1M params, expect ~20 ms/step, ~+25% steps). If the early curve
  (E1–E3) matches exp16 while steps go up, it wins; if it falls behind like
  exp20's bottleneck did, width 192 is the optimum and the width axis is
  bracketed {128, 192, 256}.
- Next iteration: exp24 = exp16 + `AE_EMB_DIM=128` (one change). After
  that: NUM_BLOCKS=6 (bounded by exp17), then a short lr warm-up to test
  3e-3. Baseline for the loop stays **exp16 (test clean 0.00533)**; default
  knobs NUM_WORKERS=4, GUARD_GAIN=0.98, BETA2=0.95, LR=2e-3, NUM_BLOCKS=4,
  BATCH_SIZE=64, N_LATENT_TOKENS=32, EMB_DIM=32, AE_EMB_DIM=192. Still 27x
  above 0.0002.

## 2026-09-18 — iteration 22: exp24 AE_EMB_DIM=128 (transformer width 192 → 128, speed direction)

No code change; `AE_EMB_DIM` knob from iteration 20.

### exp24 — exp16 + AE_EMB_DIM=128 (dense, pure recon, 4 blocks, bs 64, 4 workers, guard 0.98, beta2 0.95, lr 2e-3)

Command:
```
./run_exp.sh exp24_dense_lr2e-3_beta2_095_blocks4_ae128 MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=2e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 NUM_BLOCKS=4 AE_EMB_DIM=128 TIME_BUDGET_SEC=600
```
One change vs. exp16: transformer width 192 → 128 (heads stay 8 × 24; z
stays 32 × 32 = 1,024). Params 2,058,928 (vs 3,919,408, −47%). Removed vs.
original notebook: masked view, view term, laplacian, swd/bwd/c_swd/t_swd,
sparse middle, half the depth; guard 0.995 → 0.98; beta2 0.999 → 0.95.
Changed: transformer width 192 → 128 (narrower than notebook). Kept: full
data, RoPE jitter, bs 64, lr 2e-3, LR guard, GRAD_CLIP=0, z 1,024.

Result at the 10-min mark (22,948 steps, 14.0 epochs):

| metric | exp24 (width 128) | exp16 (width 192) |
|---|---|---|
| FINAL train clean | 0.00542 | **0.00530** |
| FINAL test clean | **0.00524** | 0.00533 |
| steps in 600 s | 22,948 | 20,979 |
| epochs in 600 s | 14.0 | 12.8 |
| ms/step (plain, median) | ~24 (range 17–31) | ~26 (range 20–32) |
| step clean at 500 / 1000 / 1640 | 0.0541 / 0.0322 / 0.0250 | 0.0542 / 0.0344 / — |
| lr after probes | 2e-3 all eight | 2e-3 all nine |

Per-epoch test clean, exp24: E1 0.02398, E2 0.01548, E3 0.01099, E4 0.00948,
E5 0.00829, E6 0.00764, E7 0.00705, E8 0.00664, E9 0.00629, E10 0.00601,
E11 0.00582, E12 0.00551, E13 0.00526, E14 0.00524. Versus exp16 (E1 0.02331,
E2 0.01742, E3 0.01266, E5 0.00908, E8 0.00699, E10 0.00615, E13 0.00533):
+3% at E1, −11% at E2, −13% at E3, −9% at E5, −5% at E8, −2% at E10, −1% at
E13. LR probes all clean (1× ratios 0.977–0.994, no scale beat 1× by 2%).

Implications:
- **Width 128 ties exp16: test clean 0.00524 vs 0.00533 (−1.7%), train
  clean 0.00542 vs 0.00530 (+2.3%).** Within run-to-run noise; not a
  meaningful win, but also the first capacity change that did not lose.
  Half the params at equal loss.
- The per-epoch curve is *ahead* of exp16 through E8 (−13% at E3) but the
  advantage decays to zero by E13: the narrower model learns faster per
  step early but saturates sooner, and the extra epoch it fits in (14 vs
  12.8) only just compensates. Width axis is now bracketed {128: 0.00524,
  192: 0.00533, 256: 0.01006}; 128–192 is a flat optimum.
- **Key finding: halving the transformer width cut params 47% but ms/step
  only ~8% (24 vs 26 ms median).** Steps in 600 s rose 9%, not the ~25%
  expected. The step is therefore not dominated by transformer FLOPs at
  this size: the fixed per-step cost (dataloader / augmentation / conv
  encoder-decoder / optimizer / Python overhead) is most of the ~24 ms.
  Model-size levers cannot buy much wall-clock; the remaining lever on
  the speed side is the pipeline itself (data loading, CUDA graphs /
  compile mode, channels_last, fused optimizer), not the architecture.
- Next iteration: profile one step to split the ~24 ms into data-wait vs
  forward/backward vs optimizer (e.g. time `next(iter)` separately, or
  run with a synthetic in-GPU batch for 500 steps and compare ms/step).
  If data-wait dominates, try NUM_WORKERS=8 / pinned memory / GPU-side
  augmentation; if GPU compute dominates, try torch.compile mode
  "max-autotune" or CUDA graphs. Baseline for the loop stays **exp16
  (test clean 0.00533)** with exp24 as an equal-loss, half-size
  alternative; default knobs unchanged (AE_EMB_DIM=192). Still 26x above
  0.0002.

## 2026-09-18 — iteration 23: prof01/prof02 per-step profile of exp16 (data vs compute vs epoch-end overhead)

Code change: `PROFILE=1` env knob in the training loop (train.py cell 50).
Default off = byte-identical behaviour. With it on: (a) each step is split
into dataloader wait / train step incl. LR guard (with `cuda.synchronize`)
/ post-step bookkeeping, printed every 500 steps; (b) epoch-end
`evaluate_multiview` and snapshots+checkpoint are timed; (c) after the
budget and after the final eval, 200 synthetic steps on the last batch are
timed with and without a per-step `.item()` sync; (d) `PROFILE_TRACE=1`
additionally runs 20 steps under `torch.profiler` and prints the kernel
table by CUDA time. No experiment on the loss this iteration.

### prof01 — exp16 config, 240 s budget, PROFILE=1

Command:
```
./run_exp.sh prof01_exp16_profile MASKED_VIEW=0 VIEW_MODE=none LAP_W=0 SWD_W=0 \
  CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=2e-3 NUM_WORKERS=4 \
  GUARD_GAIN=0.98 BETA2=0.95 NUM_BLOCKS=4 PROFILE=1 TIME_BUDGET_SEC=240
```
(8,289 steps in 240 s; test clean 0.00923 at 4 min — consistent with exp16's
curve, so the profiling hooks do not perturb training.)

Per-step split, steady state (intervals 1,000–8,000; interval 500 includes
compile warm-up at 64 ms):

| bucket | ms/step |
|---|---|
| dataloader wait (`next(iter)`) | 0.1–0.2 |
| train step incl. LR guard `.item()` + H2D copy (synced) | 20.4–25.7 (median ~23) |
| post-step: collect_metrics / log / readout | 0.3–0.7 |

Epoch-end overhead, every epoch (5 samples each):

| item | seconds |
|---|---|
| `evaluate_multiview(max_batches=20)` (train + test) | 3.53–3.56 |
| loss/recon snapshots + `save_run_snapshot` checkpoint | 0.85–1.11 |

Synthetic loop (same batch, no dataloader, no guard, after the final eval):

| variant | ms/step |
|---|---|
| `train_step` + `.item()` sync each step | 19.3 |
| `train_step`, no per-step sync | 18.9 |

### prof02 — same config, 60 s budget, PROFILE=1 PROFILE_TRACE=1

Command: as prof01 with `PROFILE_TRACE=1 TIME_BUDGET_SEC=60`.
`torch.profiler` over 20 synthetic steps: **Self CUDA time total 353.7 ms =
17.7 ms/step** of GPU kernel time vs. 18.9 ms wall → the GPU is ~94% busy
during a step. Top CUDA self-time per step (÷20): `aten::mm` 3.6 ms,
flash-attention backward 1.8 ms, cutlass bf16 GEMM 1.7 ms, `aten::addmm`
1.7 ms, three `nvjet_sm121` GEMMs 2.3 ms, **`_fused_adamw_` 1.17 ms**,
fused gelu+gelu_backward 1.16 ms, fused gelu fwd 0.66 ms, fused layernorm
0.72 ms. GEMMs ≈ 9.3 ms, elementwise/normalisation ≈ 2.5 ms, optimizer
1.2 ms, attention ≈ 2+ ms.

Implications:
- **The step is GPU-compute-bound, not data-bound and not launch-bound.**
  Dataloader wait is 0.1 ms (4 workers + pin_memory is already sufficient;
  NUM_WORKERS=8 / GPU-side augmentation would buy nothing). The two
  per-step host syncs (guard `.item()`, `collect_metrics().cpu()`) cost
  0.4 ms (19.3 vs 18.9). CUDA graphs / `mode="reduce-overhead"` cannot help
  much when the GPU is 94% busy.
- **The ~4 ms gap between the in-loop step (~23 ms) and the synthetic step
  (~19 ms)** is the LR-guard bookkeeping (`_probe_batches` deque of 4
  batches, EMA), the H2D copy of the batch, and the extra `synchronize`
  added by the profiler; it is ~17% and is a candidate for trimming, but
  not the main cost.
- **Why width 128 barely changed ms/step (exp24):** at bs 64 the GEMMs run
  at M = 64 × 64 tokens = 4,096 rows × 192 cols; these are small enough to
  sit near the kernel latency floor on GB10, so halving N/K cuts FLOPs
  without cutting time. Conversely bs 128 (exp18) doubled ms/step because
  the batch-proportional parts (elementwise, attention, memory traffic)
  do scale. Model-size changes therefore cannot buy wall-clock; only fewer
  or fused kernels per step can.
- **Cheapest real lever found: epoch-end overhead = 4.4–4.7 s per epoch,
  ≈ 57–60 s of the 600 s budget (~10%) at 12.8 epochs.** The per-epoch
  eval (3.55 s, 20 batches × 2 splits) and the per-epoch snapshot plots +
  checkpoint (0.9 s) are diagnostics, not training. Running the eval every
  4th epoch and saving snapshots only at the end would return ~50 s ≈
  +8% steps (≈ +1,700 steps; exp24's E13→E14 gain was −0.4%, so expect a
  small but free improvement, ~−1 to −3% test clean).
- Second lever: the fused AdamW at 1.17 ms/step (6%) for 3.9M params is
  oddly slow (multi_tensor_apply over many small tensors); and the fused
  gelu/layernorm triton kernels (2.5 ms, 14%) are memory-bound bf16↔fp32
  casts — `torch.compile(mode="max-autotune-no-cudagraphs")` could
  re-tile the small GEMMs and is a one-flag test.
- Next iteration: exp25 = exp16 + `EVAL_EVERY=4` / `SNAPSHOT_EVERY=0`
  knobs (add to train.py, default 1 = notebook behaviour; final eval and
  final snapshot always run). One change; expect ~22,700 steps in 600 s.
  After that: compile mode max-autotune-no-cudagraphs; then trim the
  guard's per-step cost (`_probe_batches` only when a probe is imminent).
  Baseline for the loop stays **exp16 (test clean 0.00533)**; default
  knobs unchanged. Still 27x above 0.0002.

## 2026-09-18 — iteration 24: exp25 EVAL_EVERY=4 / SNAPSHOT_EVERY=0 (cut epoch-end overhead)

Code change: `EVAL_EVERY` and `SNAPSHOT_EVERY` env knobs in the training
loop (train.py cell 50, next to `PROFILE`). `EVAL_EVERY=N` runs
`evaluate_multiview` + `readout.end_epoch` only every N-th epoch (0 = final
only; skipped epochs print a one-line `eval skipped` marker with step and
elapsed minutes). `SNAPSHOT_EVERY=N` runs loss/recon snapshot plots,
`save_run_snapshot` checkpoint and `lr_events.pt` only every N-th epoch
(0 = final only). The out-of-time epoch always evaluates and saves, so the
FINAL numbers and final checkpoint are unchanged. `completed_epoch` now uses
`epoch + 1` instead of `last_epoch + 1` so skipped epochs are counted
correctly. Default 1/1 = byte-identical notebook behaviour. Knobs listed in
`run_exp.sh` header.

### exp25 — exp16 + EVAL_EVERY=4 SNAPSHOT_EVERY=0 (dense, pure recon, 4 blocks, bs 64, 4 workers, guard 0.98, beta2 0.95, lr 2e-3)

Command:
```
./run_exp.sh exp25_dense_lr2e-3_beta2_095_blocks4_eval4_snap0 MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=2e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 NUM_BLOCKS=4 EVAL_EVERY=4 SNAPSHOT_EVERY=0 \
  TIME_BUDGET_SEC=600
```
One change vs. exp16: epoch-end eval every 4th epoch instead of every epoch,
snapshots/checkpoint only at the end. Model, data, optimizer identical.
Removed vs. original notebook: masked view, view term, laplacian,
swd/bwd/c_swd/t_swd, sparse middle, half the depth, per-epoch eval and
per-epoch snapshots (diagnostics only; training unaffected); guard 0.995 →
0.98; beta2 0.999 → 0.95. Kept: full data, RoPE jitter, bs 64, lr 2e-3, LR
guard, GRAD_CLIP=0, width 192, z 1,024.

Result at the 10-min mark (22,375 steps, 13.6 epochs; 4 evals total):

| metric | exp25 (eval/4, snap end) | exp16 (eval+snap every epoch) |
|---|---|---|
| FINAL train clean | 0.00535 | **0.00530** |
| FINAL test clean | **0.00511** | 0.00533 |
| steps in 600 s | 22,375 | 20,979 |
| epochs in 600 s | 13.6 | 12.8 |
| wall min/epoch (skipped-eval epochs) | 0.69–0.75 | ~0.78 |
| test clean at E4 / E8 / E12 | 0.01042 / 0.00709 / 0.00554 | (E3 0.01266, E5 0.00908) / 0.00699 / (E13 0.00533) |
| lr after probes | 2e-3 all (1× ratios 0.981–1.006) | 2e-3 all nine |

Implications:
- **Test clean 0.00511 vs 0.00533 (−4.1%), the best 10-minute test clean so
  far; train clean 0.00535 vs 0.00530 (+0.9%, noise).** Gain is the
  predicted +1,400 steps (+6.7%) from removing ~50 s of diagnostics; the
  per-epoch curve is unchanged (E8 0.00709 vs 0.00699), so this is pure
  wall-clock, not a training change. Matches the iteration-23 estimate of
  −1 to −3% (slightly better).
- The remaining per-eval cost is 3 × 3.5 s ≈ 11 s (EVAL_EVERY=4 leaves 3
  interior evals). EVAL_EVERY=0 would return another ~11 s ≈ +400 steps
  (~+2%) but loses the per-epoch curve entirely; keep 4 as the loop
  default trade-off (the curve is what lets us attribute changes).
- Cosmetic: with skipped evals the readout's `@E<n>` best-epoch label and
  `gain/1ep` count evals, not epochs (E4 shows `@E1`, gain vs. init). Not
  worth fixing; the `E<n>/300 | test clean` lines carry the true epoch.
- **New loop baseline: exp25 (test clean 0.00511, train clean 0.00535).**
  All future exp16-family runs should carry `EVAL_EVERY=4 SNAPSHOT_EVERY=0`.
  Still 26x above 0.0002; the compute-bound step (~19 ms synthetic, ~23 ms
  in-loop) is the remaining wall-clock lever.
- Next iteration: exp26 = exp25 + `torch.compile(mode="max-autotune-no-cudagraphs")`
  (add a `COMPILE_MODE` knob; default = current mode). The kernel table
  showed ~9 ms of small GEMMs near the latency floor and 2.5 ms of
  memory-bound triton elementwise; autotune is a one-flag test of whether
  re-tiling helps at M=4,096. Watch compile warm-up time — if autotune eats
  >60 s of the budget it loses by construction. After that: trim the LR
  guard's per-step `_probe_batches` bookkeeping (~4 ms/step gap between
  in-loop and synthetic step).

## 2026-09-18 — iteration 25: exp26 COMPILE_MODE=max-autotune-no-cudagraphs (loses: +85 s warm-up, no per-step gain, worse per-epoch curve)

Code change: `COMPILE_MODE` env knob in train.py cell 44 — passed as
`torch.compile(train_step, mode=...)`; empty or `default` = notebook
behaviour (`mode=None`). Listed in `run_exp.sh` header. No other change.

### exp26 — exp25 + COMPILE_MODE=max-autotune-no-cudagraphs

Command:
```
./run_exp.sh exp26_dense_lr2e-3_beta2_095_blocks4_eval4_snap0_maxautotune MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=2e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 NUM_BLOCKS=4 EVAL_EVERY=4 SNAPSHOT_EVERY=0 \
  COMPILE_MODE=max-autotune-no-cudagraphs TIME_BUDGET_SEC=600
```
Removed/kept vs. original notebook: identical to exp25 (pure recon, dense,
4 blocks, guard 0.98, beta2 0.95, eval/4, snapshots at end).

Result at the 10-min mark (19,190 steps, 11.7 epochs):

| metric | exp26 (max-autotune-no-cudagraphs) | exp25 (default compile) |
|---|---|---|
| FINAL train clean | 0.00631 | **0.00535** |
| FINAL test clean | 0.00626 | **0.00511** |
| steps in 600 s | 19,190 | 22,375 |
| epoch 1 wall (incl. compile warm-up) | 2.42 min | 0.99 min |
| first-500-steps ms/step | 233 | 63 |
| steady ms/step (500-step windows) | 21–30, median ~25 | 22–27, median ~25 |
| test clean at E4 / E8 / E12 | 0.01218 / 0.00789 / 0.00626 | 0.01042 / 0.00709 / 0.00554 |
| train clean at step 500 / 1000 / 1640 | 0.0594 / 0.0372 / 0.0280 | 0.0542 / 0.0344 / 0.0249 |
| lr after probes | 2e-3 all eight (1× ratios 0.958–0.992) | 2e-3 all nine |

Observations:
- **Test clean 0.00626 vs 0.00511 (+22.5%); clear loss.** Two independent
  causes: (a) autotune warm-up cost **~85 s** of the 600 s budget
  (−3,185 steps, −14%); (b) steady-state step time did **not** improve
  (median ~25 ms both runs), so there was nothing to recoup it with.
- **Inductor disabled GEMM template autotuning on GB10:** log line
  `Not enough SMs to use max_autotune_gemm mode` — `torch._inductor.utils.is_big_gpu`
  requires ≥ 68 SMs (3080 class); GB10 has 48. The 10 `AUTOTUNE addmm(...)`
  blocks that did run each had only **2 choices** (ATen/cuBLAS variants,
  0.07–0.35 s each), so the "re-tile the small GEMMs with Triton templates"
  hypothesis from iteration 23 was never actually tested; only the Triton
  pointwise/reduction kernels were autotuned (that is where the 85 s went).
  The GEMM shapes seen: M=8,192 rows (64 imgs × 128 enc+dec tokens),
  K/N ∈ {192, 384, 576, 768}.
- **The per-epoch curve is also worse (E8 0.00789 vs 0.00709, +11%; train
  loss already +10% at step 500)**, with the LR guard holding 2e-3 in both
  runs. Same seed/data/optimizer, so this is a numerics change from the
  autotuned pointwise kernels (different fusion/reduction choices in the
  fused layernorm/gelu paths), not noise — exp16 vs exp25 agreed within
  1.5% at E8. Not investigated further because the mode loses on wall
  clock regardless.
- Forcing the GEMM templates on (monkeypatch `is_big_gpu` → True, or
  `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON`) would add still more
  warm-up on top of 85 s and requires Triton GEMMs to beat cuBLAS at
  M=8,192/K=192 on 48 SMs — unlikely to net positive inside a 10-minute
  box. Parked; not worth an iteration.
- **COMPILE_MODE stays at default; exp25 remains the loop baseline
  (test clean 0.00511, train clean 0.00535).** Still 26x above 0.0002.
- Next iteration: the remaining wall-clock lever from the profile is the
  ~4 ms/step gap between the in-loop step (~23 ms) and the synthetic
  `train_step` (~19 ms): LR-guard `_probe_batches` deque bookkeeping +
  per-step `collect_metrics().cpu()` sync + H2D copy. Add a knob to only
  retain probe batches when a probe is imminent (last N steps of an epoch)
  and to log metrics every 500 steps only; target ~+15% steps. After that:
  `non_blocking=True` H2D with pinned memory if not already, and
  `torch.optim.AdamW(fused=True)` parameter-flattening (1.17 ms/step).

## 2026-09-18 — iteration 26: exp27 GUARD_PROBES=0 (drop the LR guard's epoch-end probes; wins, new baseline)

Code change: `GUARD_PROBES` env knob next to `GUARD_GAIN` (train.py cell 48).
`GUARD_PROBES=0` keeps the guard's loss EMA and non-finite check but never
fires an LR probe and no longer retains the 4-batch `_probe_batches` deque.
Default 1 = notebook behaviour. Listed in `run_exp.sh` header.

Why this experiment: in exp25's log every epoch-end probe (9 of them, all
choosing scale=1 with 1× ratios 0.98–1.01) inflated the surrounding 500-step
window from ~24 ms/step to 28–37 ms/step (≈ +3 s each, ~27 s total), and each
probe's clone/restore of weights+grads invalidated dynamo's guards on the
compiled optimizer frame (`Cache line invalidated because L['args'][5][0]
got deallocated`) — exp25 hit `recompile_limit (8)` on `maybe_fallback` at
13:28, after the 8th probe. The guard has not lowered lr in any exp16-family
run, so the probes were pure cost.

### exp27 — exp25 + GUARD_PROBES=0 (dense, pure recon, 4 blocks, bs 64, 4 workers, beta2 0.95, lr 2e-3, eval/4, snap end)

Command:
```
./run_exp.sh exp27_dense_lr2e-3_beta2_095_blocks4_eval4_snap0_noprobe MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=2e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 NUM_BLOCKS=4 EVAL_EVERY=4 SNAPSHOT_EVERY=0 \
  GUARD_PROBES=0 TIME_BUDGET_SEC=600
```
One change vs. exp25: no LR probes. Model, data, optimizer, lr identical
(lr fixed at 2e-3 for the whole run, which is what the probes produced anyway).
Removed vs. original notebook: masked view, view term, laplacian,
swd/bwd/c_swd/t_swd, sparse middle, half the depth, per-epoch eval and
snapshots, **LR-guard probes** (EMA/non-finite check kept); guard gain 0.98,
beta2 0.95. Kept: full data, RoPE jitter, bs 64, lr 2e-3, GRAD_CLIP=0,
width 192, z 1,024.

Result at the 10-min mark (24,006 steps, 14.6 epochs; 4 evals):

| metric | exp27 (no probes) | exp25 (probes) |
|---|---|---|
| FINAL train clean | **0.00500** | 0.00535 |
| FINAL test clean | **0.00487** | 0.00511 |
| steps in 600 s | 24,006 | 22,375 |
| epochs in 600 s | 14.6 | 13.6 |
| test clean at E4 / E8 / E12 | 0.01042 / 0.00709 / 0.00554 | 0.01042 / 0.00709 / 0.00554 |
| train loss at step 500 / 1640 / 8200 / 16400 | 0.05416 / 0.02488 / 0.00964 / 0.00632 | identical |
| ms/step, 500-step windows containing an epoch end | 21–25 (one 32) | 28–37 |
| dynamo recompile_limit warnings | 0 | 1 (after 8th probe) |
| wall at end of E13 | 9.00 min | ~9.2 min |

Implications:
- **Test clean 0.00487 vs 0.00511 (−4.7%), train clean 0.00500 vs 0.00535
  (−6.5%); best 10-minute result so far.** The per-epoch and per-step
  curves are bit-for-bit identical to exp25 (same seed, same data order,
  lr never changed by the probes), so the whole gain is the +1,631 steps
  (+7.3%) bought back from probe time and optimizer-frame recompiles.
- Cumulative wall-clock trims since exp16: eval/4 + snap-end (+1,400 steps)
  and no probes (+1,631 steps) → 24,006 steps vs 20,979 (+14.4%) and test
  clean 0.00533 → 0.00487 (−8.6%) with no training change.
- Steady-state ms/step is now 21–25 in every window (median ~23), i.e. the
  in-loop step is within ~2–4 ms of the synthetic 19 ms floor; remaining
  host-side items are the per-step `.item()` + `collect_metrics().cpu()`
  syncs (~0.4 ms) and readout tensor ops. Little left to trim outside
  train_step itself.
- The guard's safety role is now only the non-finite check. If a future lr
  change (e.g. 3e-3) needs the probes back, `GUARD_PROBES=1` restores them.
- **New loop baseline: exp27 (test clean 0.00487, train clean 0.00500).**
  All future exp16-family runs carry `EVAL_EVERY=4 SNAPSHOT_EVERY=0
  GUARD_PROBES=0`. Still 24x above 0.0002; the remaining lever is the
  GPU-bound step (~19 ms synthetic, ~9 ms GEMMs near the latency floor).
- Next iteration: with the probes gone, lr is now a pure constant, so a
  wall-clock cosine schedule (`LR_SCHED=cosine`, decaying to LR_MIN_FRAC
  over the last part of the 600 s) is the cheapest remaining training-side
  test — earlier schedule attempts were confounded by the guard rescaling
  `_sched_mult`. After that: `torch.optim.AdamW` foreach vs fused timing,
  and a bf16 `autocast` check on the decoder's final projection (largest
  elementwise/memory-bound kernels in the profile).

## 2026-09-18 — iteration 27: exp28 LR 4e-3 + 300-step warm-up, no decay (loses; lr axis exhausted upward)

Why not `LR_SCHED=cosine` as iteration 26 suggested: exp05 (full-budget
cosine, −31%) and exp14 (anneal over last 25%, 0.00569 vs 0.00568 wash) were
*not* confounded by the guard — every probe in both runs kept scale 1, so
`_sched_mult` stayed 1.0 and the schedule ran exactly as designed. Repeating
it would redo logged work. The one lr idea never run is the iteration-13
follow-up: 4e-3 with a warm-up, testing whether exp15's collapse was only the
no-warm-up early blow-up.

Code change (train.py): `_scheduled_lr` returned `lr` after warm-up only via
`max(0, t-1)/(1-1)` when `DECAY_START_FRAC=1`, i.e. ZeroDivisionError. Added
an explicit `DECAY_START_FRAC >= 1 → return lr` branch (warm-up only, constant
lr afterwards). No behaviour change for any previously used value.

Note: the first launch of this run (`nohup ... &` inside the tool shell) was
killed with the shell and never created a log file; ~10 min of the iteration
budget lost. Launch `run_exp.sh` directly as a background task instead.

### exp28 — exp27 + LR=4e-3, linear warm-up 300 steps, then constant

Command:
```
./run_exp.sh exp28_dense_lr4e-3_warmup300_blocks4_eval4_snap0_noprobe MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=4e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 NUM_BLOCKS=4 EVAL_EVERY=4 SNAPSHOT_EVERY=0 \
  GUARD_PROBES=0 LR_SCHED=cosine WARMUP_STEPS=300 DECAY_START_FRAC=1 LR_MIN_FRAC=1 TIME_BUDGET_SEC=600
```
One change vs. exp27: lr 2e-3 constant → 4e-3 after a 300-step linear
warm-up (constant after). Removed/kept relative to the notebook: as exp27.

Result (run still in flight at the iteration deadline; numbers through E8,
test eval at 6.0 min — final numbers in `artifacts/logs/exp28_*.log`):

| metric | exp28 (4e-3, warm-up) | exp27 (2e-3) |
|---|---|---|
| train clean at step 500 / 1640 | 0.05602 / 0.02590 | 0.05416 / 0.02488 |
| train clean at step 3280 / 4920 | 0.01822 / 0.01427 | 0.01821 / 0.01351 |
| train clean at step 6560 / 8200 | 0.01280 / **0.01100** | 0.01053 / **0.00964** |
| test clean at E4 / E8 | **0.01264** / **0.00895** | **0.01042** / **0.00709** |
| wall at E4 end | 3.06 min | ~2.95 min |
| ms/step (steady) | 21–24 | 21–25 |

Implications:
- **The warm-up does keep lr 4e-3 alive** (exp15 without warm-up was a dead
  mean-colour predictor at step 500, clean 0.068; exp28 is 0.056 at step 500
  and keeps descending). So exp15's collapse was the early blow-up, not a
  hard stability limit at 4e-3.
- **But 4e-3 is slower per step from ~step 4,000 on**: tied with exp27 at
  step 3,280, then +5.6% at 4,920, +21.6% at 6,560, +14% at 8,200 and +21%
  on E4 test clean. Training is noise-limited at this lr, consistent with
  exp14's "loss moves at a rate ∝ lr" only up to ~2e-3. Unless the second
  half of the run reverses that (check the log's FINAL line; not expected),
  **2e-3 stays the lr and the lr axis is exhausted in both directions**:
  decay loses (exp05/exp14), 4e-3 loses (exp28), 4e-3 without warm-up dies
  (exp15).
- exp27 remains the loop baseline (test clean 0.00487). Still 24x above
  0.0002; the gap will not close via optimizer hyperparameters at this
  model/step cost.
- Next iteration: back to the step-cost lever. Candidates, in order:
  (1) `torch.optim.AdamW(fused=True)` vs current foreach (1.17 ms/step in
  the iteration-23 profile, ~5%); (2) channels_last / bf16 autocast on the
  decoder's final projection and pixel-space MSE (largest memory-bound
  kernels); (3) BATCH_SIZE=128 at lr 2e-3 now that the guard is passive —
  earlier bs tests were at other lr/beta2 settings.

## 2026-09-18 — iteration 28: exp29 SYNC_EVERY=50 (drop per-step host syncs; loses on wall clock, inconclusive on cause)

exp28 postscript: the exp28 container (relaunched by iteration 27 as a
background task) ran to completion and exited at 07:24:59, but its host log
stopped receiving lines after E9 (6.44 min; `docker logs` showed E10 at
7.10 min with train clean 0.00817 at step 16,400 vs exp27's 0.00632, +29%).
Its FINAL numbers are lost. Conclusion from iteration 27 stands: lr 4e-3
loses at every checkpoint; exp27 stays baseline.

Code change (train.py): `SYNC_EVERY` env knob (next to `GUARD_PROBES`).
`train_step_with_lr_guard` returns straight after `train_step` unless
`state["step"] % SYNC_EVERY == 0`, and the loop only calls
`collect_metrics()` / `log_metrics()` on those steps. So with SYNC_EVERY=50
the only per-step host↔GPU sync (guard `.item()` + `collect_metrics().cpu()`)
happens once every 50 steps; `readout` accumulates on-device as before.
Default 1 = notebook behaviour. Listed in `run_exp.sh` header.

Hypothesis: the per-step sync serialises CPU kernel launch with GPU
execution; letting the CPU queue ahead should recover part of the ~2–4 ms
gap between the in-loop step (21–25 ms) and the synthetic 19 ms step.

### exp29 — exp27 + SYNC_EVERY=50

Command:
```
./run_exp.sh exp29_dense_lr2e-3_blocks4_eval4_snap0_noprobe_sync50 MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=2e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 NUM_BLOCKS=4 EVAL_EVERY=4 SNAPSHOT_EVERY=0 \
  GUARD_PROBES=0 SYNC_EVERY=50 TIME_BUDGET_SEC=600
```
One change vs. exp27: host sync every 50 steps instead of every step.
Removed/kept relative to the notebook: as exp27, plus per-step loss
logging thinned to every 50th step (guard non-finite check likewise).

Result at the 10-min mark (22,767 steps, 13.9 epochs):

| metric | exp29 (sync/50) | exp27 (sync/1) |
|---|---|---|
| FINAL train clean | 0.00529 | **0.00500** |
| FINAL test clean | 0.00504 | **0.00487** |
| steps in 600 s | 22,767 | 24,006 |
| test clean at E4 / E8 / E12 | 0.01042 / 0.00709 / 0.00554 | identical |
| train clean at step 1640 / 8200 / 16400 | 0.02488 / 0.00964 / 0.00632 | identical |
| wall at E3 / E5 / E10 / E13 end | 2.60 / 4.09 / 7.22 / 9.37 min | 2.29 / 3.58 / 6.84 / 9.00 min |
| ms/step windows (500-step) | 19–33; 16 windows ≥ 28, three at 19 | 20–32; 8 windows ≥ 28, none < 20 |
| GPU util / SM clock (nvidia-smi samples at 2.6 and 5.3 min) | 55–67 % / 2,470 MHz (max 3,003) | not sampled |
| trainer main process CPU | 104 % (single core pegged); 4 workers at ~13 % each | not sampled |

Implications:
- **Loses: −5.2% steps → test clean 0.00504 vs 0.00487 (+3.5%).** The
  per-step/per-epoch loss curves are bit-identical (the change touches no
  math), so the whole difference is wall clock.
- The slowdown is concentrated in minutes 1–4 (windows at 28–33 ms while
  exp27 was at 21–25); from step ~9,700 to ~11,500 exp29 ran 19–21 ms/step,
  *faster* than any exp27 window, then drifted back to 21–28. Either the
  removal of the sync makes step time bimodal (CPU runs ahead until the
  launch queue / caching allocator stalls it — main process pegged at 104 %
  CPU with GPU at 55–67 % is consistent with a host-launch-bound loop, which
  contradicts the iteration-23 profile's 94 % GPU-busy figure taken *with*
  the sync), or the GPU/CPU was shared with something else during the first
  minutes (the exp28 container had exited 20 s before launch; host load avg
  was 2.0). One run cannot separate these.
- **Not adopted; exp27 stays the loop baseline (test clean 0.00487).**
  `SYNC_EVERY` stays in the code at default 1 (no behaviour change).
- Next iteration: a 2-minute A/B (`TIME_BUDGET_SEC=120`, back-to-back,
  SYNC_EVERY=1 vs 50, compare steps at 120 s) settles whether the sync
  removal is neutral, bimodal or harmful for ~7 min total, and should be
  paired with `nvidia-smi dmon`-style sampling inside the container to see
  whether GPU util is really ~60 % in the steady state. If util is ~60 %
  with the sync too, the GB10 step is host-launch-bound after all and the
  lever is fewer kernels per step (CUDA graphs via
  `COMPILE_MODE=reduce-overhead`, previously dismissed on the 94 % figure),
  not faster kernels.

## 2026-09-18 — iteration 29: exp30 COMPILE_MODE=reduce-overhead (CUDA graphs) → test clean 0.00445, new best (−8.6% vs exp27)

Why this instead of the 2-minute SYNC_EVERY A/B suggested by iteration 28:
the A/B only diagnoses; `reduce-overhead` tests the "host-launch-bound"
hypothesis *and* the lever in one 10-minute run. If the step is really
94 % GPU-busy (iteration-23 profile) CUDA graphs buy ≤1 ms/step; if the
in-loop step is launch-bound (exp29's pegged main process, 55–67 % util)
they buy the whole 2–10 ms gap between the 19 ms synthetic step and the
20–33 ms in-loop step. The earlier dismissal was reasoning from the
synthetic profile, never a measurement.

No code change: `COMPILE_MODE` already existed (iteration 25). GPU-util
sampling inside the container was not possible (`docker exec` needs an
interactive permission prompt this session cannot answer), so the
evidence is the run's own `ms/step` column.

### exp30 — exp27 + COMPILE_MODE=reduce-overhead

Command:
```
./run_exp.sh exp30_dense_lr2e-3_blocks4_eval4_snap0_noprobe_cudagraphs MASKED_VIEW=0 VIEW_MODE=none \
  LAP_W=0 SWD_W=0 CSWD_W=0 TSWD_W=0 BWD_W=0 VIEW_W=0 SPARSE_MIDDLE=0 LR=2e-3 \
  NUM_WORKERS=4 GUARD_GAIN=0.98 BETA2=0.95 NUM_BLOCKS=4 EVAL_EVERY=4 SNAPSHOT_EVERY=0 \
  GUARD_PROBES=0 COMPILE_MODE=reduce-overhead TIME_BUDGET_SEC=600
```
One change vs. exp27: `torch.compile(mode="reduce-overhead")` on
`train_step` (CUDA-graph replay of the whole fwd/bwd/optimizer step).
Removed/kept relative to the notebook: as exp27 (all non-clean loss terms
zeroed, masked view off, dense middle, 4 blocks, lr 2e-3 constant, beta2
0.95, eval every 4 epochs, no snapshots, no LR-guard probes).

Result at the 10-min mark (29,077 steps, 17.7 epochs):

| metric | exp30 (cudagraphs) | exp27 (baseline) |
|---|---|---|
| FINAL train clean | **0.00464** | 0.00500 |
| FINAL test clean | **0.00445** | 0.00487 |
| steps in 600 s | **29,077** | 24,006 |
| test clean at E4 / E8 / E12 / E16 | 0.01042 / 0.00709 / 0.00554 / 0.00471 | 0.01042 / 0.00709 / 0.00554 / (not reached) |
| train clean at step 1640 / 8200 / 16400 / 24600 | 0.02488 / 0.00964 / 0.00632 / 0.00491 | 0.02488 / 0.00964 / 0.00632 / — |
| wall at E3 / E5 / E10 / E15 end | 1.92 / 3.04 / 5.77 / 8.48 min | 2.29 / 3.58 / 6.84 / 9.61 min |
| epoch wall (steady) | 0.53–0.54 min | 0.66–0.78 min |
| ms/step, 500-step windows (excl. compile window) | 44 windows at 19, 26 at 20; none above 20 | 20–32, bimodal |
| img/s (steady) | 3,270–3,360 | 2,160–3,160 |
| cudagraph fallback warnings ("skipping cudagraphs") | none | n/a |

Implications:
- **Wins: +21 % steps in the same 600 s → test clean 0.00445 (−8.6 %),
  train clean 0.00464 (−7.2 %).** Per-step and per-epoch losses are
  bit-identical to exp27, so the gain is purely wall clock, exactly as
  with GUARD_PROBES=0 and unlike SYNC_EVERY=50.
- **The in-loop step was launch-bound after all.** With graphs the step
  sits at the synthetic floor (19–20 ms) with zero variance; the bimodal
  21–33 ms of exp27/exp29 was CPU launch overhead (the ~150 kernels/step
  of a 4-block fwd+bwd+fused-AdamW on a single pegged core), not GPU
  contention. The iteration-23 "94 % busy" figure was correct for the
  profiler's synthetic step, and wrong as a description of the training
  loop. SYNC_EVERY=50 made things worse because removing the sync let the
  CPU run further ahead into the same launch bottleneck, not because
  syncs were expensive.
- Compile warm-up is unchanged (first 500-step window 60 ms/step, E1 ends
  at 0.87 min vs 0.91), so graphs do not cost warm-up time here.
- **New loop baseline: exp30 (test clean 0.00445, train clean 0.00464).**
  All future exp16-family runs add `COMPILE_MODE=reduce-overhead`. Still
  22x above 0.0002.
- With the launch floor gone, the step is now genuinely GPU-bound at
  ~19 ms (≈9 ms GEMMs near the latency floor, ≈2 ms attention, ≈2.5 ms
  elementwise, 1.2 ms AdamW). Levers that were previously masked by
  launch overhead are now worth re-testing under graphs, in order:
  (1) `BATCH_SIZE=128` — exp18 doubled ms/step at bs 128, but part of
  that was launch-bound noise; under graphs bs 128 may cost <2x and the
  GEMMs (M = 8,192 rows) leave the latency floor; (2) `NUM_BLOCKS=3`
  (fewer kernels *and* fewer FLOPs; exp-family blocks tests were done
  pre-graphs); (3) bf16 autocast on the decoder head + pixel MSE.
  `SYNC_EVERY>1` may also now be neutral-to-positive under graphs (the
  guard `.item()` forces a graph-replay wait each step) — a cheap
  follow-up once the bs question is settled.

## 2026-09-18 — OWNER NOTE (post-overnight review): the 0.0045 floor was latent collapse

Diagnosis from the repo owner after reviewing exp01–exp30:

- The ~0.0045 test-clean plateau reached by exp27/exp30 was **latent collapse**,
  not the latent-bottleneck capacity ceiling hypothesized throughout this log.
- **Cause: high lr in combination with a high view-alignment term weight.**
  These two interact; the lr brackets recorded above (2e-3 best, 4e-3 dead)
  were measured under collapse dynamics and need re-bracketing once collapse
  is controlled.
- Consequence for the ledger: comparisons between runs that plateaued near
  0.0045–0.0055 are comparisons between collapsed latents and should not be
  read as evidence about capacity (N_LATENT_TOKENS, EMB_DIM, AE_EMB_DIM).
- Going forward every run must log latent health at eval (per-channel z std,
  effective rank, dead-channel fraction) — see RESEARCH.md "Owner findings".
