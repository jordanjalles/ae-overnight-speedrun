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

# %% cell 0
from pathlib import Path

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import STL10
import time


# DO NOT use CUDA_LAUNCH_BLOCKING=1 during normal training.
# It forces synchronization after CUDA launches and hurts throughput.
os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

# Allow fast Tensor Core matmuls (TF32 on supported hardware)
torch.set_float32_matmul_precision("high")


data_root = Path(os.environ["ML_DATASETS"])

train_ds = STL10(
    root=data_root,
    split="train+unlabeled",
    download=True,
    transform=transforms.ToTensor(),
)

test_ds = STL10(
    root=data_root,
    split="test",
    download=True,
    transform=transforms.ToTensor(),
)
DATA_SEED = 42

train_generator = torch.Generator()
train_generator.manual_seed(DATA_SEED)

train_dataloader = DataLoader(
    train_ds,
    shuffle=True,
    batch_size=int(os.environ.get("BATCH_SIZE", 64)),  # overnight loop knob
    drop_last=True,
    generator=train_generator,
    num_workers=int(os.environ.get("NUM_WORKERS", 0)),  # overnight loop knob
    pin_memory=int(os.environ.get("NUM_WORKERS", 0)) > 0,
    persistent_workers=int(os.environ.get("NUM_WORKERS", 0)) > 0,
)

test_dataloader = DataLoader(
    test_ds,
    batch_size=256,
    shuffle=False,
)

# %% cell 1
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
print(os.getcwd())

# %% cell 2
24*8

# %% cell 3
#hyperparameters

#model width and latent tokens
ae_emb_dim = 192
n_ae_latent_tokens = 64

num_blocks = 8 #dinov2 12
n_heads = 8
head_size = 24 #must be divisible by 4 for 2D rope
               #emb_dim = n_heads × head_dim <---------default ViT ratio
#z dimensions
emb_dim = 32
n_latent_tokens = 32

#Training settings
RUN_NAME = "ae_7M_jr_smt_hpreg_spike_guard_lr2e-3" #high pressure regularization
use_masked_view = True #whether or not to train masked prediction
view_mode = "clean_target" #clean-masked view alignment loss mode


#learning
lr = 2e-3
layer_scale_gamma = 1e-4 #at init

#acts on x_hat  
laplacian_weight = 1.0

#z regularizers (and invariance)
swd_weight = 1.0  #flattened latent swd
c_swd_weight = 1.0   #channel wise swd
t_swd_weight = 1.0  #token wise swd
bwd_weight = 1.0   #batch wise swd for every dim
view_weight = 0.1 #0.001 was too small


TRACK_Z_DIAGNOSTICS = False


# --- overnight loop env overrides (RESEARCH.md: 10-min from-scratch runs) ---
# Unset env vars leave the notebook defaults above untouched.
def _env(name, default, cast=float):
    return cast(os.environ[name]) if name in os.environ else default

RUN_NAME = _env("RUN_NAME", RUN_NAME, str)
use_masked_view = bool(_env("MASKED_VIEW", int(use_masked_view), int))
view_mode = _env("VIEW_MODE", view_mode if use_masked_view else "none", str)
lr = _env("LR", lr)
laplacian_weight = _env("LAP_W", laplacian_weight)
swd_weight = _env("SWD_W", swd_weight)
c_swd_weight = _env("CSWD_W", c_swd_weight)
t_swd_weight = _env("TSWD_W", t_swd_weight)
bwd_weight = _env("BWD_W", bwd_weight)
view_weight = _env("VIEW_W", view_weight)
# z bottleneck size (model-size change; log it for phase-2 restoration)
emb_dim = _env("EMB_DIM", emb_dim, int)
n_latent_tokens = _env("N_LATENT_TOKENS", n_latent_tokens, int)
# Transformer depth (notebook 8). Needs >= shallow+final (2); with SPARSE_MIDDLE=1 use > 2.
num_blocks = _env("NUM_BLOCKS", num_blocks, int)
# Transformer width (notebook 192 = 8 heads x 24). EMB_DIM is only the z channel
# width; AE_EMB_DIM / N_HEADS / HEAD_SIZE are the real compute/capacity levers.
ae_emb_dim = _env("AE_EMB_DIM", ae_emb_dim, int)
n_heads = _env("N_HEADS", n_heads, int)
head_size = _env("HEAD_SIZE", head_size, int)  # must be divisible by 4 (2D RoPE)
# Wall-clock budget for the training loop (compile warm-up included);
# the loop stops, runs the final eval, and prints a FINAL line.
TIME_BUDGET_SEC = _env("TIME_BUDGET_SEC", 600.0)
# Global grad-norm clip applied before optimizer.step(); 0 = off (notebook).
GRAD_CLIP = _env("GRAD_CLIP", 0.0)
# AdamW betas; notebook defaults (0.9, 0.999).
BETA1 = _env("BETA1", 0.9)
BETA2 = _env("BETA2", 0.999)
print("env overrides:", {k: v for k, v in os.environ.items() if k in {
    "RUN_NAME", "MASKED_VIEW", "VIEW_MODE", "LR", "LAP_W", "SWD_W", "CSWD_W",
    "TSWD_W", "BWD_W", "VIEW_W", "TIME_BUDGET_SEC", "EPOCHS", "SPARSE_MIDDLE",
    "EMB_DIM", "N_LATENT_TOKENS", "LR_SCHED", "WARMUP_STEPS", "LR_MIN_FRAC",
    "BATCH_SIZE", "NUM_WORKERS", "GUARD_GAIN", "GRAD_CLIP", "BETA1", "BETA2", "DECAY_START_FRAC",
    "NUM_BLOCKS", "AE_EMB_DIM", "N_HEADS", "HEAD_SIZE"}})


patch_size = 12
batches_per_epoch = len(iter(train_dataloader))
b, c, h, w = next(iter(train_dataloader))[0].shape
pc = w//patch_size #patch columns
pr = h//patch_size #patch rows
patch_rows = pr
patch_cols = pc
total_patches = pc * pr
latent_dim = n_latent_tokens*emb_dim
patch_dim = patch_size * patch_size * c
image_dim = h*w*c

# %% cell 4
# ================= Dimension Summary =================

image_dim = c * h * w
patch_dim = c * patch_size * patch_size

ae_latent_dim = n_ae_latent_tokens * ae_emb_dim
latent_dim = n_latent_tokens * emb_dim

print("========== DIMENSION SUMMARY ==========\n")

print("IMAGE / PATCHES")
print(f"image shape:              {c} × {h} × {w}")
print(f"image dim:                {image_dim:,}")
print(f"patch size:               {patch_size} × {patch_size}")
print(f"patch grid:               {pr} × {pc}")
print(f"patch tokens:             {total_patches}")
print(f"patch dim:                {patch_dim}")
print(f"patch → AE width:         {patch_dim} → {ae_emb_dim}   "
      f"({patch_dim / ae_emb_dim:.2f}× compression)")

print("\nAE WORKSPACE")
print(f"AE token count:           {n_ae_latent_tokens}")
print(f"AE width:                 {ae_emb_dim}")
print(f"AE workspace dim:         {ae_latent_dim:,}")
print(f"attention:                {n_heads} heads × {head_size} dim "
      f"= {n_heads * head_size}")

print("\nLATENT z")
print(f"z token count:            {n_latent_tokens}")
print(f"z channel dim:            {emb_dim}")
print(f"z shape:                  ({n_latent_tokens}, {emb_dim})")
print(f"z total dim:              {latent_dim:,}")

print("\nCOMPRESSION")
print(f"patch tokens → AE tokens: {total_patches} → {n_ae_latent_tokens}   "
      f"({total_patches / n_ae_latent_tokens:.2f}×)")
print(f"AE tokens → z tokens:     {n_ae_latent_tokens} → {n_latent_tokens}   "
      f"({n_ae_latent_tokens / n_latent_tokens:.2f}×)")
print(f"AE width → z width:       {ae_emb_dim} → {emb_dim}   "
      f"({ae_emb_dim / emb_dim:.2f}×)")
print(f"AE workspace → z:         {ae_latent_dim:,} → {latent_dim:,}   "
      f"({ae_latent_dim / latent_dim:.2f}×)")
print(f"image → z:                {image_dim:,} → {latent_dim:,}   "
      f"({image_dim / latent_dim:.2f}×)")

# %% cell 5
#a dataset sample
import matplotlib.pyplot as plt

image, label = train_ds[1]   # image shape: (C, H, W)

plt.imshow(image.permute(1, 2, 0))  # convert to (H, W, C)
plt.axis("off")
plt.show()

# %% cell 6
print('num-batches:', len(iter(train_dataloader)), '| batch-shape:', next(iter(train_dataloader))[0].shape)

# %% cell 7
import torch
from torch import nn


def apply_2d_rope(q, k, n_latent_tokens, rope):
    assert q.shape == k.shape
    B, H, T, D = q.shape
    assert D % 4 == 0

    # Tables are either (P, D/4) or (B, P, D/4).
    n_patches = rope[0].shape[-2]

    # Infer the actual latent count for dense or sparse passes.
    n_latent_tokens = T - n_patches
    assert n_latent_tokens >= 0

    tables = []
    for table in rope:
        table = table.to(device=q.device, dtype=q.dtype)
        if table.ndim == 3:
            table = table.unsqueeze(1)  # (B, 1, P, D/4)
        tables.append(table)

    row_cos, row_sin, col_cos, col_sin = tables

    def rotate_pairs(x, cos, sin):
        x = x.reshape(B, H, n_patches, D // 4, 2)
        a, b = x.unbind(dim=-1)
        return torch.stack(
            (a * cos - b * sin, a * sin + b * cos),
            dim=-1,
        ).flatten(-2)

    def rotate_tensor(x):
        latents = x[:, :, :n_latent_tokens]
        patches = x[:, :, n_latent_tokens:]

        row, col = patches.chunk(2, dim=-1)
        row = rotate_pairs(row, row_cos, row_sin)
        col = rotate_pairs(col, col_cos, col_sin)

        patches = torch.cat((row, col), dim=-1)
        return torch.cat((latents, patches), dim=2)

    return rotate_tensor(q), rotate_tensor(k)

# %% cell 8
#pre-norm multi-head attention
class MHA(nn.Module):
    def __init__(self, 
                 n_heads, 
                 head_size, 
                 emb_dim, 
                 n_latent_tokens
                ):
        super().__init__()

        self.n_heads = n_heads
        self.head_size = head_size
        self.emb_dim = emb_dim
        self.n_latent_tokens = n_latent_tokens

        self.norm = nn.LayerNorm(emb_dim)
        self.QKV = nn.Linear(emb_dim, 3*n_heads * head_size)

        self.out_proj = nn.Linear(n_heads * head_size, emb_dim)

        self.gamma = nn.Parameter(torch.full( (emb_dim,), layer_scale_gamma))

    def forward(self, x, rope):
        residual = x
        
        
        #useful for shape viewing/permuting below
        B, T, C = x.shape
        nh = self.n_heads
        hs = self.head_size

        #normalize
        x = self.norm(x)
        
        qkv = self.QKV(x) # (B, T, C) -> ( B, T, 3 * n_heads *head_size )

        #slice qkv aprt
        q = qkv[:, :, : nh * hs ] # ( B, T, n_heads * head_size )
        k = qkv[:, :, nh * hs : 2 * nh * hs ] # ( B, T, n_heads * head_size )
        v = qkv[:, :, 2 * nh * hs :] # ( B, T, n_heads * head_size )


        
        #reshape (B, T, n_heads*headsize) -> (B, n_heads, T, headsize)
        q = q.view(B, T, nh, hs).permute(0, 2, 1, 3)
        k = k.view(B, T, nh, hs).permute(0, 2, 1, 3)
        v = v.view(B, T, nh, hs).permute(0, 2, 1, 3)

        q, k = apply_2d_rope(
            q,
            k,
            n_latent_tokens=self.n_latent_tokens,
            rope=rope,
        )

        #q @ k.t / head_size **0.5
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )

        out = out.permute(0, 2, 1, 3).flatten(-2) # (B, T, H*hs)

        out = residual + self.gamma * self.out_proj(out) # (B T C)


        return out

# %% cell 9
class MLP(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.gamma = nn.Parameter(torch.full((emb_dim,), layer_scale_gamma))
        
        self.net = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim*4),
            nn.GELU(),
            nn.Linear(emb_dim*4, emb_dim)
        )

    def forward(self, x):
        return x + self.gamma * self.net(x)

# %% cell 10
#transformer+MLP block
class Block(nn.Module):
    def __init__(self, 
                 n_heads, 
                 head_size, 
                 emb_dim,
                 n_latent_tokens):
        super().__init__()

        self.MHA = MHA(n_heads, head_size, emb_dim, n_latent_tokens)
        self.MLP = MLP(emb_dim)

    def forward(self, x, rope):

        x = self.MHA(x, rope)
        
        x = self.MLP(x)

        return x

# %% cell 11
class PatchEmbed(nn.Module):
    def __init__(self, patch_size, c, emb_dim):
        super().__init__()

        self.patch_size = patch_size

        self.embed = nn.Conv2d(in_channels = c,
                             out_channels = emb_dim,
                             kernel_size = patch_size,
                             stride = patch_size,
                             padding=0)


    def forward(self, x):
        B, C, H, W = x.shape
        patch_rows = H//patch_size
        patch_cols = W//patch_size

        
        out = self.embed(x) # (B, C, H, W) -> (B, emb_dim, pr, pc)
        out = out.flatten(-2).permute(0, 2, 1) # (B, T, C)

        return out, patch_rows, patch_cols

# %% cell 12
class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size, c, emb_dim):
        super().__init__()

        self.patch_size = patch_size
        self.c = c
        self.emb_dim = emb_dim

        self.to_pixels = nn.Linear(emb_dim, c * patch_size**2)

    def forward(self, x, patch_rows, patch_cols):
        B, T, C = x.shape
        p = self.patch_size
        c = self.c

        x = self.to_pixels(x) # BTC -> B, T, c*p**2

        # Split token position and patch contents
        # (B, pr, pc, c, patch, patch)

        x = x.reshape(B,  patch_rows, patch_cols, c, p, p)

        # Interleave patch-grid and within-patch coordinates
        # (B, c, pr, patch, pc, patch)
        x = x.permute(0, 3, 1, 4, 2, 5)

        # Join them into image height and width
        # (B, c, pr*p, pc*p) 
        out = x.reshape(B, c, patch_rows*p, patch_cols*p)

        return out

# %% cell 13
#define core transformer model 
import math

class CoreTransformer(nn.Module):
    def __init__(self, 
                 emb_dim, 
                 num_blocks, 
                 n_heads, 
                 head_size, 
                 n_latent_tokens
                ):
        
        super().__init__()

        self.n_latent_tokens = n_latent_tokens

        self.blocks = nn.ModuleList([
            Block(n_heads, head_size, emb_dim, n_latent_tokens)
                for _ in range(num_blocks)
        ])
        
        self.norm = nn.LayerNorm(emb_dim)

        #ROPE BUFFER
        pair_idx = torch.arange(head_size//4, dtype=torch.float32)
        inv_freq = 100 ** (-pair_idx / (head_size // 4))

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("row_sin_cache", torch.empty(0), persistent=False)
        self.register_buffer("row_cos_cache", torch.empty(0), persistent=False)
        self.register_buffer("col_sin_cache", torch.empty(0), persistent=False)
        self.register_buffer("col_cos_cache", torch.empty(0), persistent=False)

        self.rope_cache_grid = None
 
    def forward(self, x, patch_rows, patch_cols, keep_indices=None):
        rope = self.get_rope_cache(patch_rows, patch_cols)

        if keep_indices is not None:
            rope = tuple(table[keep_indices] for table in rope)

        for block in self.blocks:
            x = block(x, rope)

        return self.norm(x)

    def get_rope_cache(self, patch_rows, patch_cols):
        grid = (patch_rows, patch_cols) #current layout

        # Rebuild during training to sample fresh jitter.
        # Rebuild during eval to avoid reusing jittered training tables.
        if self.training or self.rope_cache_grid != grid:
            row_pos = (
                2 * torch.arange(
                    patch_rows, device=self.inv_freq.device, dtype=torch.float32
                ) / max(patch_rows - 1, 1) - float(patch_rows > 1)
            ).repeat_interleave(patch_cols)
            
            col_pos = (
                2 * torch.arange(
                    patch_cols, device=self.inv_freq.device, dtype=torch.float32
                ) / max(patch_cols - 1, 1) - float(patch_cols > 1)
            ).repeat(patch_rows)

            
    
            scale = 1.0

            if self.training:
                # Uniform in log-space: shrinking and enlarging are balanced.
                log_scale = torch.empty(
                    (), device=self.inv_freq.device
                ).uniform_(-math.log(2), math.log(2))
            
                scale = log_scale.exp()
            
            row_angles = (
                2 * math.pi * scale
                * row_pos[:, None] * self.inv_freq[None, :]
            )
            col_angles = (
                2 * math.pi * scale
                * col_pos[:, None] * self.inv_freq[None, :]
            )
    
            self.row_cos_cache = row_angles.cos()
            self.row_sin_cache = row_angles.sin()
            self.col_cos_cache = col_angles.cos()
            self.col_sin_cache = col_angles.sin()
    
            # Training tables must not be reused during evaluation.
            self.rope_cache_grid = None if self.training else grid
        return (
            self.row_cos_cache,
            self.row_sin_cache,
            self.col_cos_cache,
            self.col_sin_cache,
        )

# %% cell 14
class SparseMiddleTransformer(CoreTransformer):
    """
    Shallow blocks → sparse middle blocks → fusion → final blocks.

    Uses the existing CoreTransformer's RoPE setup, including jitter.
    Input-masked patches remain absent throughout this core.
    """

    def __init__(
        self,
        emb_dim,
        num_blocks,
        n_heads,
        head_size,
        n_latent_tokens,
        keep_ratio=0.25,
        shallow_blocks=1,
        final_blocks=1,
    ):
        super().__init__(
            emb_dim=emb_dim,
            num_blocks=num_blocks,
            n_heads=n_heads,
            head_size=head_size,
            n_latent_tokens=n_latent_tokens,
        )

        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1].")
        if shallow_blocks < 1 or final_blocks < 1:
            raise ValueError("Need at least one shallow and final block.")
        # Zero middle blocks is allowed (overnight loop NUM_BLOCKS=2): the
        # middle loop is empty and fusion(shallow, shallow) is identity at init.
        if shallow_blocks + final_blocks > num_blocks:
            raise ValueError("Need num_blocks >= shallow + final blocks.")

        self.keep_ratio = keep_ratio
        self.shallow_blocks = shallow_blocks
        self.middle_end = num_blocks - final_blocks

        # Set False for full-token fine-tuning.
        self.sparse_middle = True

        # Fuse shallow and deep features: 2C → C.
        self.fusion = nn.Linear(2 * emb_dim, emb_dim)

        # Start with equal contributions from both paths.
        with torch.no_grad():
            eye = torch.eye(emb_dim)
            self.fusion.weight.copy_(
                torch.cat((0.5 * eye, 0.5 * eye), dim=1)
            )
            self.fusion.bias.zero_()

    @staticmethod
    def gather_tokens(x, indices):
        return x.gather(
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]),
        )

    @staticmethod
    def gather_rope(rope, indices):
        selected = []
        for table in rope:
            if table.ndim == 2:
                # Shared full-grid table.
                selected.append(table[indices])
            else:
                # Already selected per image by input masking.
                selected.append(
                    table.gather(
                        dim=1,
                        index=indices.unsqueeze(-1).expand(
                            -1, -1, table.shape[-1],
                        ),
                    )
                )
        return tuple(selected)

    @staticmethod
    def sample_indices(batch, count, keep, device):
        # Sort selected indices to preserve original token order.
        return (
            torch.rand(batch, count, device=device)
            .topk(keep, dim=1, largest=False)
            .indices.sort(dim=1)
            .values
        )

    def forward(self, x, patch_rows, patch_cols, keep_indices=None):
        rope = self.get_rope_cache(patch_rows, patch_cols)

        # Input masking: x already contains only visible patches.
        if keep_indices is not None:
            rope = self.gather_rope(rope, keep_indices)

        B, T, C = x.shape
        L = self.n_latent_tokens
        P = T - L
        assert P == rope[0].shape[-2]

        # Shallow path sees all available tokens.
        for i in range(self.shallow_blocks):
            x = self.blocks[i](x, rope)

        shallow = x

        use_sparse = (
            self.training
            and self.sparse_middle
            and self.keep_ratio < 1
        )

        if use_sparse:
            # Fixed counts. At least one survives in each nonempty group.
            keep_latents = min(L, max(1, int(L * self.keep_ratio)))
            keep_patches = min(P, max(1, int(P * self.keep_ratio)))

            latent_ids = self.sample_indices(
                B, L, keep_latents, x.device,
            )
            patch_ids = self.sample_indices(
                B, P, keep_patches, x.device,
            )

            # Latents remain the prefix; patches follow.
            token_ids = torch.cat(
                (latent_ids, L + patch_ids), dim=1,
            )

            deep = self.gather_tokens(shallow, token_ids)
            middle_rope = self.gather_rope(rope, patch_ids)
        else:
            deep = shallow
            middle_rope = rope

        for i in range(self.shallow_blocks, self.middle_end):
            deep = self.blocks[i](deep, middle_rope)

        if use_sparse:
            # Restore deep features to their original positions.
            deep = torch.zeros_like(shallow).scatter(
                dim=1,
                index=token_ids.unsqueeze(-1).expand(-1, -1, C),
                src=deep,
            )

        x = self.fusion(torch.cat((shallow, deep), dim=-1))

        for i in range(self.middle_end, len(self.blocks)):
            x = self.blocks[i](x, rope)

        return self.norm(x)

# %% cell 15
#define encoder model 
    
class Encoder(nn.Module):
    def __init__(self, 
                 patch_size, 
                 in_channels, 
                 ae_emb_dim,
                 emb_dim, 
                 n_ae_latent_tokens,
                 n_latent_tokens,
                 num_blocks, 
                 n_heads, 
                 head_size, 
                ):
        
        super().__init__()


        self.ae_emb_dim = ae_emb_dim
        self.emb_dim = emb_dim

        self.n_latent_tokens = n_latent_tokens
        self.n_ae_latent_tokens = n_ae_latent_tokens

        self.latent_tokens = nn.Parameter(torch.randn((1, n_ae_latent_tokens, ae_emb_dim)))

        self.embed = PatchEmbed(patch_size, in_channels, ae_emb_dim)

        self.sparse_middle_transformer = SparseMiddleTransformer(emb_dim = ae_emb_dim, 
                                                 num_blocks = num_blocks, 
                                                 n_heads = n_heads, 
                                                 head_size = head_size, 
                                                 n_latent_tokens = n_ae_latent_tokens
                                                )

        self.out_norm = nn.LayerNorm(ae_emb_dim)
        self.token_out_proj = nn.Linear(n_ae_latent_tokens, n_latent_tokens, bias=False)
        self.emb_out_proj = nn.Linear(ae_emb_dim, emb_dim, bias=False)
        
    def forward(self, image, keep_indices=None):
        
        patches, patch_rows, patch_cols = self.embed(image)

        if keep_indices is not None:
            indices = keep_indices.unsqueeze(-1).expand(
                -1, -1, patches.shape[-1],
            )
            patches = patches.gather(dim=1, index=indices)
        
        # cat latent tokens with patch embed
        B = patches.shape[0]
        
        latents = self.latent_tokens.expand(B, -1, -1)
        x = torch.cat([latents, patches], dim=1)
        
        x = self.sparse_middle_transformer(
            x, patch_rows, patch_cols,
            keep_indices=keep_indices,
        )

        latents = x[:, :self.n_latent_tokens]
        latents = self.out_norm(latents)
        z = self.emb_out_proj(latents)

        return z

# %% cell 16
# define decoder model

class Decoder(nn.Module):
    def __init__(
        self,
        patch_size,
        out_channels,
        ae_emb_dim,
        emb_dim,
        n_ae_latent_tokens,
        n_latent_tokens,
        num_blocks,
        n_heads,
        head_size,
    ):
        super().__init__()

        self.ae_emb_dim = ae_emb_dim
        self.emb_dim = emb_dim

        self.n_ae_latent_tokens = n_ae_latent_tokens
        self.n_latent_tokens = n_latent_tokens

        self.latent_slots = nn.Parameter(
            torch.randn(1, n_ae_latent_tokens, ae_emb_dim) * 0.02
        )

        self.emb_in_proj = nn.Linear(
            emb_dim,
            ae_emb_dim,
            bias=False,
        )

        self.token_in_proj = nn.Linear(
            n_latent_tokens,
            n_ae_latent_tokens,
            bias=False,
        )

        self.learned_patch_token = nn.Parameter(
            torch.randn(1, 1, ae_emb_dim)
        )

        self.sparse_middle_transformer = SparseMiddleTransformer(
            emb_dim=ae_emb_dim,
            num_blocks=num_blocks,
            n_heads=n_heads,
            head_size=head_size,
            n_latent_tokens=n_ae_latent_tokens,
        )

        self.un_embed = PatchUnEmbed(
            patch_size=patch_size,
            c=out_channels,
            emb_dim=ae_emb_dim,
        )

        self.in_norm = nn.LayerNorm(ae_emb_dim)
        self.out_norm = nn.LayerNorm(ae_emb_dim)
        
    def forward(self, z):

        
        # Restore latent embedding width:
        # (B, n_latent_tokens, emb_dim)
        # -> (B, n_latent_tokens, ae_emb_dim)
        latents = self.emb_in_proj(z)
        
        # token expansion with zero-padding:
        # (B, n_latent_tokens, ae_emb_dim)
        # -> (B, n_ae_latent_tokens, ae_emb_dim)
        latents = F.pad(
            latents,
            (
                0, 0,
                0, self.n_ae_latent_tokens - self.n_latent_tokens,
            ),
        )
        
        latents = latents + self.latent_slots
    
        B = latents.shape[0]
    
        patches = self.learned_patch_token.expand(
            B,
            patch_rows * patch_cols,
            -1,
        )

        x = torch.cat([latents, patches], dim=1)

        x = self.in_norm(x)
        
        x = self.sparse_middle_transformer(
            x,
            patch_rows,
            patch_cols,
        )

        x = self.out_norm(x)

        patch_embeddings = x[:, self.n_ae_latent_tokens:]

        images = torch.sigmoid(
            self.un_embed(
                patch_embeddings,
                patch_rows,
                patch_cols,
            )
        )

        return images

# %% cell 17
def sliced_wasserstein_distance(
    z,
    prior,
    n_slices=32,
    bulk_sigma=1.0,       # smaller = stronger tail discount; None = unweighted
):
    assert z.shape == prior.shape
    if bulk_sigma is not None and bulk_sigma <= 0:
        raise ValueError("bulk_sigma must be positive or None")

    z = z.flatten(1)                         # (B, D)
    prior = prior.detach().flatten(1)        # fixed target
    B, D = z.shape

    slices = F.normalize(
        torch.randn(D, n_slices, device=z.device, dtype=z.dtype),
        dim=0,
    )                                       # (D, S)

    z_slices = torch.sort(z @ slices, dim=0).values.float()
    p_slices = torch.sort(prior @ slices, dim=0).values.float()

    squared_error = (z_slices - p_slices).square()

    if bulk_sigma is None:
        loss = squared_error.mean()
    else:
        log_weights = -0.5 * (p_slices / bulk_sigma).square()

        # Normalize across batch ranks, separately for each slice.
        # Softmax stably computes exp(log_weights) / sum(exp(log_weights)).
        weights = torch.softmax(log_weights, dim=0)

        loss = (weights * squared_error).sum(dim=0).mean()

    # Square the aggregate loss to damp pressure at low mismatch.
    return loss.square()

# %% cell 18
def basis_wasserstein_distance(z, prior, bulk_sigma=1.0):
    assert z.shape == prior.shape
    assert z.ndim == 3, "Expected z shaped (B, T, C)"

    if bulk_sigma is not None and bulk_sigma <= 0:
        raise ValueError("bulk_sigma must be positive or None")

    prior = prior.detach()

    def axis_loss(source, target, dim):
        source = torch.sort(source, dim=dim).values.float()
        target = torch.sort(target, dim=dim).values.float()

        squared_error = (source - target).square()

        if bulk_sigma is None:
            return squared_error.mean()

        log_weights = -0.5 * (target / bulk_sigma).square()
        weights = torch.softmax(log_weights, dim=dim)

        return (weights * squared_error).sum(dim=dim).mean()

    loss_b = axis_loss(z, prior, dim=0)   # across images, per (t, c)
    loss_t = axis_loss(z, prior, dim=1)   # across tokens, per (b, c)
    loss_c = axis_loss(z, prior, dim=2)   # across channels, per (b, t)
    loss_e = axis_loss(
        z.flatten(1), prior.flatten(1), dim=1
    )                                    # across T*C, per image

    loss = loss_b # + 0.1 * (loss_t + loss_c + loss_e) / 3
    
    # Square the aggregate loss to damp pressure at low mismatch.
    return  loss.square()

# %% cell 19
def token_sliced_wasserstein_distance(
    z,
    prior,
    n_slices=32,
    bulk_sigma=1.0,
):
    """
    Token-space SWD.

    z/prior: (B, T, C)

    Reshapes to (B*C, T), so each channel across the token sequence
    is one sample, and random slices operate across the T token dimensions.

    This directly pressures token combinations/correlations
    toward the isotropic Gaussian prior.
    """
    assert z.shape == prior.shape
    assert z.ndim == 3, "Expected z shaped (B, T, C)"

    if bulk_sigma is not None and bulk_sigma <= 0:
        raise ValueError("bulk_sigma must be positive or None")

    B, T, C = z.shape

    # Each channel's values across T tokens becomes one T-dimensional observation.
    z = z.permute(0, 2, 1).reshape(B * C, T)
    prior = prior.detach().permute(0, 2, 1).reshape(B * C, T)

    # Random directions through token space.
    slices = F.normalize(
        torch.randn(
            T,
            n_slices,
            device=z.device,
            dtype=z.dtype,
        ),
        dim=0,
    )  # (T, S)

    # Project each observation onto random token-space directions.
    z_slices = torch.sort(
        z @ slices,
        dim=0,
    ).values.float()

    p_slices = torch.sort(
        prior @ slices,
        dim=0,
    ).values.float()

    squared_error = (z_slices - p_slices).square()

    if bulk_sigma is None:
        loss = squared_error.mean()

    else:
        log_weights = -0.5 * (p_slices / bulk_sigma).square()

        # Normalize across the B*C observations for each slice.
        weights = torch.softmax(
            log_weights,
            dim=0,
        )

        loss = (
            weights * squared_error
        ).sum(dim=0).mean()

    return loss.square()

# %% cell 20
def channel_sliced_wasserstein_distance(
    z,
    prior,
    n_slices=32,
    bulk_sigma=1.0,
):
    """
    Channel-space SWD.

    z/prior: (B, T, C)

    Reshapes to (B*T, C), so each token embedding is one sample
    and random slices operate across the C channel dimensions.

    This directly pressures channel combinations/correlations
    toward the isotropic Gaussian prior.
    """
    assert z.shape == prior.shape
    assert z.ndim == 3, "Expected z shaped (B, T, C)"

    if bulk_sigma is not None and bulk_sigma <= 0:
        raise ValueError("bulk_sigma must be positive or None")

    B, T, C = z.shape

    # Each token embedding becomes one C-dimensional observation.
    z = z.reshape(B * T, C)
    prior = prior.detach().reshape(B * T, C)

    # Random directions through channel space.
    slices = F.normalize(
        torch.randn(
            C,
            n_slices,
            device=z.device,
            dtype=z.dtype,
        ),
        dim=0,
    )  # (C, S)

    # Project each token embedding onto each random channel direction.
    z_slices = torch.sort(
        z @ slices,
        dim=0,
    ).values.float()

    p_slices = torch.sort(
        prior @ slices,
        dim=0,
    ).values.float()

    squared_error = (z_slices - p_slices).square()

    if bulk_sigma is None:
        loss = squared_error.mean()

    else:
        log_weights = -0.5 * (p_slices / bulk_sigma).square()

        # Normalize across the B*T observations for each slice.
        weights = torch.softmax(
            log_weights,
            dim=0,
        )

        loss = (
            weights * squared_error
        ).sum(dim=0).mean()

    # Same low-mismatch damping behavior as your other Wasserstein losses.
    return loss.square()

# %% cell 21
def laplacian(x):
    C = x.shape[1]

    kernel = torch.tensor(
        [[0, -1,  0],
         [-1, 4, -1],
         [0, -1,  0]],
        device=x.device,
        dtype=x.dtype,
    )

    kernel = kernel[None, None].repeat(C, 1, 1, 1)

    return F.conv2d(x, kernel, padding=1, groups=C)

# %% cell 22
# helper: render reconstruction
@torch.no_grad()
def render_reconstruction(
    masked=False,
    show=True,
):
    features, _ = next(iter(test_dataloader))
    features = features[:8].to(device)

    if masked:
        mask, keep_indices = make_mask(
            features,
            mask_ratio=0.5,
            return_keep_indices=True,
        )

        z = encoder(
            features,
            keep_indices=keep_indices,
        )

        pixel_mask = make_pixel_mask(
            mask,
            features,
        )

        display_features = (
            features * (1.0 - pixel_mask)
        )

    else:
        z = encoder(features)
        display_features = features

    x_hat = decoder(z)

    n = features.shape[0]

    fig, axes = plt.subplots(
        2,
        n,
        figsize=(2 * n, 4),
    )

    for i in range(n):
        axes[0, i].imshow(
            display_features[i]
            .float()
            .cpu()
            .permute(1, 2, 0)
            .clamp(0, 1)
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(
            x_hat[i]
            .float()
            .cpu()
            .permute(1, 2, 0)
            .clamp(0, 1)
        )
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel(
        "masked input" if masked else "input",
        rotation=0,
        labelpad=45,
        va="center",
    )

    axes[1, 0].set_ylabel(
        "recon",
        rotation=0,
        labelpad=35,
        va="center",
    )

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

# %% cell 23
#helper: def reconstruct images
@torch.no_grad()
def reconstruct_images(images, close_fig = False):
    
    x_hat = decoder(encoder(images))

    x = images[0].detach().cpu().permute(1, 2, 0)
    recon = x_hat[0].detach().cpu().permute(1, 2, 0)
    #recon = x_hat[0,0]

    fig, axs = plt.subplots(1, 2, figsize = (6,3))

    axs[0].imshow(x)
    axs[0].set_title('input')
    axs[0].axis('off')

    
    axs[1].imshow(recon)
    axs[1].set_title('reconstruction')
    axs[1].axis('off')

    if close_fig:
        plt.close(fig)

# %% cell 24
#helper: def plot_training_snapshot
import os
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def plot_training_snapshot(
    recon_targets,
    step,
    total_steps,
    save_path=None,
    show=False,
):
    """
    Plot four fixed reconstruction examples.

    Row 1: original targets
    Row 2: model reconstructions

    Assumes:
        recon_targets: (4, C, H, W)
        encoder and decoder exist in the outer scope
    """
    if recon_targets.ndim != 4:
        raise ValueError(
            f"Expected recon_targets shaped (4, C, H, W), "
            f"got {recon_targets.shape}"
        )

    if recon_targets.shape[0] != 4:
        raise ValueError(
            f"Expected exactly 4 reconstruction targets, "
            f"got {recon_targets.shape[0]}"
        )

    # Infer the model device rather than relying on a global `device`.
    model_device = next(encoder.parameters()).device
    targets = recon_targets.to(model_device)

    encoder_was_training = encoder.training
    decoder_was_training = decoder.training

    encoder.eval()
    decoder.eval()

    try:
        reconstructions = decoder(encoder(targets))
    finally:
        encoder.train(encoder_was_training)
        decoder.train(decoder_was_training)

    targets = targets.detach().float().cpu()
    reconstructions = reconstructions.detach().float().cpu()

    if reconstructions.shape != targets.shape:
        raise ValueError(
            "Target and reconstruction shapes do not match:\n"
            f"targets:         {targets.shape}\n"
            f"reconstructions: {reconstructions.shape}"
        )

    def prepare_for_imshow(image):
        """
        Convert:
            (1, H, W) -> (H, W)
            (3, H, W) -> (H, W, 3)
        """
        if image.ndim != 3:
            raise ValueError(
                f"Expected individual image shaped (C, H, W), got {image.shape}"
            )

        channels = image.shape[0]

        if channels == 1:
            return image.squeeze(0).clamp(0, 1).numpy()

        if channels in (3, 4):
            return image.permute(1, 2, 0).clamp(0, 1).numpy()

        raise ValueError(f"Cannot display image with {channels} channels")

    # Per-example reconstruction error for a compact progress signal.
    mse = (
        (targets - reconstructions)
        .square()
        .flatten(start_dim=1)
        .mean(dim=1)
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12, 6),
        gridspec_kw={
            "hspace": 0.08,
            "wspace": 0.08,
        },
    )

    fig.suptitle(
        f"Autoencoder Training — Step {step:,} / {total_steps:,}",
        fontsize=15,
        y=0.97,
    )

    for i in range(4):
        target = prepare_for_imshow(targets[i])
        reconstruction = prepare_for_imshow(reconstructions[i])

        # Top row: original
        axes[0, i].imshow(
            target,
            cmap="gray" if targets.shape[1] == 1 else None,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axes[0, i].set_title(
            f"Sample {i + 1}\nMSE: {mse[i].item():.4f}",
            fontsize=10,
        )
        axes[0, i].axis("off")

        # Bottom row: reconstruction
        axes[1, i].imshow(
            reconstruction,
            cmap="gray" if reconstructions.shape[1] == 1 else None,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axes[1, i].axis("off")

    # Shared row labels avoid repeating eight titles.
    fig.text(
        0.025,
        0.69,
        "Target",
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
    )
    fig.text(
        0.025,
        0.29,
        "Reconstruction",
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
    )

    fig.subplots_adjust(
        top=0.86,
        bottom=0.04,
        left=0.06,
        right=0.99,
    )

    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=120,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    plt.close(fig)

# %% cell 25
# get split losses

@torch.inference_mode()
def get_split_loss(split):
    encoder_was_training = encoder.training
    decoder_was_training = decoder.training

    encoder.eval()
    decoder.eval()

    if split == "test":
        dataloader = test_dataloader
        max_batches = None

    elif split == "train":
        dataloader = train_dataloader
        max_batches = 20

    else:
        raise ValueError(f"Unknown split: {split}")

    masked_squared_error = torch.zeros((), device=device)
    clean_squared_error = torch.zeros((), device=device)
    num_elements = 0

    try:
        for batch_index, (x, _) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            x = x.to(device)

            x_hat_masked = decoder(
                encoder(x, make_mask(x))
            )

            masked_squared_error += F.mse_loss(
                x_hat_masked,
                x,
                reduction="sum",
            )

            x_hat_clean = decoder(
                encoder(x)
            )

            clean_squared_error += F.mse_loss(
                x_hat_clean,
                x,
                reduction="sum",
            )

            num_elements += x.numel()

    finally:
        encoder.train(encoder_was_training)
        decoder.train(decoder_was_training)

    loss_masked = masked_squared_error / num_elements
    loss_clean = clean_squared_error / num_elements

    return loss_masked, loss_clean

# %% cell 26
#log metrics
from collections import defaultdict
from copy import deepcopy

if "logs" not in globals():
    logs = defaultdict(list)


def clear_logs():
    logs.clear()


def log_metrics(**metrics):
    for name, value in metrics.items():
        if value is None:
            continue

        if torch.is_tensor(value):
            value = value.detach().item()

        logs[name].append(value)

def collect_metrics(metrics):
    names = []
    tensors = []
    numbers = {}

    for name, value in metrics.items():
        if value is None:
            continue

        if torch.is_tensor(value):
            names.append(name)
            tensors.append(value.detach().reshape(()))
        else:
            numbers[name] = value

    if tensors:
        values = (
            torch.stack(tensors)
            .float()
            .cpu()
            .tolist()
        )

        numbers.update(zip(names, values))

    return numbers


def get_metrics():
    return {
        name: history[-1]
        for name, history in logs.items()
        if history
    }

def get_metrics_lists():
    return {
        name: history.copy()
        for name, history in logs.items()
        if history
    }

# %% cell 27
## plot decoded images from list of seeds
@torch.no_grad()
def plot_constructions(seeds, patch_rows=None, patch_cols=None):

    constructs = decoder(seeds.to(device)).cpu()

    n = constructs.shape[0]

    fig, axes = plt.subplots(1, n, figsize=(n * 2, 4))

    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        img = constructs[i].permute(1, 2, 0).float().clamp(0, 1)
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# %% cell 28
# plot objective losses
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def moving_average(values, window=100):
    values = np.asarray(values)
    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )

def relative_change(values, eps=1e-12):
    values = np.asarray(
        values,
        dtype=float,
    )

    if values.size == 0:
        return values

    baseline = values[0]

    # Entire series is effectively zero
    if np.all(np.abs(values) < eps):
        return np.zeros_like(values)

    # Relative change is undefined if window starts at zero
    if abs(baseline) < eps:
        return np.full_like(
            values,
            np.nan,
        )

    return 100 * (
        values / baseline - 1
    )

def plot_objective_losses(
    window=1000,
    start=0,
    end=None,
    relative=False,
    show=True,
):
    history = get_metrics_lists()

    fig, ax = plt.subplots(figsize=(10, 5))

    clean = moving_average(
        history["clean_term"],
        window,
    )[start:end]

    laplacian = moving_average(
        history["laplacian_term"],
        window,
    )[start:end]

    masked = moving_average(
        history["masked_term"],
        window,
    )[start:end]

    view = moving_average(
        history["view_term"],
        window,
    )[start:end]

    if relative:
        clean = relative_change(clean)
        laplacian = relative_change(laplacian)
        masked = relative_change(masked)
        view = relative_change(view)

    ax.plot(
        clean,
        label="clean recon",
    )

    ax.plot(
        laplacian,
        label="Laplacian",
    )

    # Uncomment for masked runs
    ax.plot(
        masked,
        label="masked prediction",
    )

    # Uncomment if you want view alignment here
    # ax.plot(
    #     view,
    #     label="view invariance",
    # )

    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _: (
                f"start\n({start:,})"
                if abs(x) < 1
                else f"+{x/1000:.0f}k"
            )
        )
    )

    ax.set_xlabel("Training step")

    if relative:
        ax.set_ylabel("Change from window start (%)")
        ax.set_title(
            f"objective losses — relative change, "
            f"{window}-step moving average"
        )
        ax.axhline(
            0,
            linewidth=1,
            alpha=0.3,
        )
    else:
        ax.set_ylabel("Smoothed loss")
        ax.set_title(
            f"objective losses — "
            f"{window}-step moving average"
        )

    ax.legend()
    ax.grid(alpha=0.3)

    if show:
        plt.show()

    return fig

# %% cell 29
def plot_regularization_losses(
    window=1000,
    start=0,
    end=None,
    relative=False,
    show=True,
):
    history = get_metrics_lists()

    series = {
        "view alignment": "view_term",
        "SWD": "swd_term",
        "BWD": "bwd_term",
        "C_SWD": "c_swd_term",
        "T_SWD": "t_swd_term",
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    for label, key in series.items():
        values = moving_average(
            history[key],
            window,
        )[start:end]

        if relative:
            values = relative_change(values)

        ax.plot(values, label=label)

    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _: (
                f"start\n({start:,})"
                if abs(x) < 1
                else f"+{x/1000:.0f}k"
            )
        )
    )

    ax.set_xlabel("Training step")

    if relative:
        ax.set_ylabel("Change from window start (%)")
        ax.set_title(
            f"regularization losses — relative change, "
            f"{window}-step moving average"
        )
        ax.axhline(
            0,
            linewidth=1,
            alpha=0.3,
        )
    else:
        ax.set_ylabel("Smoothed loss")
        ax.set_title(
            f"regularization losses — "
            f"{window}-step moving average"
        )

    ax.legend()
    ax.grid(alpha=0.3)

    if show:
        plt.show()

    return fig

# %% cell 30
def save_reconstruction_snapshot(masked=False):
    fig = render_reconstruction(
        masked=masked,
        show=False,
    )

    filename = (
        "latest_recon_masked.png"
        if masked
        else "latest_recon_clean.png"
    )

    fig.savefig(
        preview_dir / filename,
        dpi=130,
        bbox_inches="tight",
    )

    plt.close(fig)

# %% cell 31
def save_loss_snapshots(
    window_full=1000,
    window_recent=500,
    recent_steps=20_000,
):
    history = get_metrics_lists()
    n_steps = len(history["clean_term"])

    fig = plot_objective_losses(
        window=window_full,
        start=0,
        show=False,
    )
    fig.savefig(
        preview_dir / "latest_objective_full.png",
        dpi=130,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig = plot_regularization_losses(
        window=window_full,
        start=0,
        show=False,
    )
    fig.savefig(
        preview_dir / "latest_regularization_full.png",
        dpi=130,
        bbox_inches="tight",
    )
    plt.close(fig)

    recent_start = max(0, n_steps - recent_steps)

    fig = plot_objective_losses(
        window=window_recent,
        start=recent_start,
        relative=True,
        show=False,
    )
    fig.savefig(
        preview_dir / "latest_objective_recent.png",
        dpi=130,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig = plot_regularization_losses(
        window=window_recent,
        start=recent_start,
        relative=True,
        show=False,
    )
    fig.savefig(
        preview_dir / "latest_regularization_recent.png",
        dpi=130,
        bbox_inches="tight",
    )
    plt.close(fig)

# %% cell 32
#Helper: MULTIVIEW EVALUATION ##

@torch.no_grad()
def evaluate_multiview(max_batches=20, seed=1234):
    encoder_was_training = encoder.training
    decoder_was_training = decoder.training

    encoder.eval()
    decoder.eval()

    eval_device = next(encoder.parameters()).device
    cuda_devices = (
        [eval_device.index if eval_device.index is not None
         else torch.cuda.current_device()]
        if eval_device.type == "cuda" else []
    )

    def evaluate_split(loader):
        clean_sum = 0.0
        clean_count = 0
        masked_sum = 0.0
        masked_count = 0

        for batch_index, (images, _) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            images = images.to(device)
            B, channels, H, W = images.shape
            assert H % pr == 0 and W % pc == 0

            mask, keep_indices = make_mask(
                images,
                mask_ratio=0.5,
                return_keep_indices=True,
            )

            pixel_mask = (
                mask.reshape(B, 1, pr, pc)
                .repeat_interleave(H // pr, dim=2)
                .repeat_interleave(W // pc, dim=3)
            ).to(images.dtype)

            clean = decoder(encoder(images))
            masked = decoder(
                encoder(images, keep_indices=keep_indices)
            )

            clean_sum += F.mse_loss(
                clean, images, reduction="sum"
            ).item()
            clean_count += images.numel()

            masked_errors = F.mse_loss(
                masked, images, reduction="none"
            )
            masked_sum += (masked_errors * pixel_mask).sum().item()
            masked_count += pixel_mask.sum().item() * channels

        if clean_count == 0 or masked_count == 0:
            raise ValueError("Evaluation needs images and at least one masked pixel.")

        return clean_sum / clean_count, masked_sum / masked_count

    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(seed)

            test_clean, test_missing = evaluate_split(test_dataloader)
            train_clean, train_missing = evaluate_split(train_dataloader)
    finally:
        encoder.train(encoder_was_training)
        decoder.train(decoder_was_training)

    return {
        "test clean": test_clean,
        "train clean": train_clean,
        "test missing": test_missing,
        "train missing": train_missing,
    }

# %% cell 33
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import json
import os
import re
import torch


# ------------------------------------------------------------------
# Storage
# ------------------------------------------------------------------

PROJECT_ROOT = Path(os.environ["ML_CHECKPOINTS"]).parent
RUN_ROOT = PROJECT_ROOT / "training_runs" / "ae"
CHECKPOINT_ROOT = Path(os.environ["ML_CHECKPOINTS"]) / "ae"

RUN_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)


# Preserve state if this setup cell is rerun.
if "logs" not in globals():
    logs = defaultdict(list)

if "runs" not in globals():
    runs = {}

if "_active_run" not in globals():
    _active_run = None

if "current_run_name" not in globals():
    current_run_name = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def sanitize_run_name(name):
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return name.strip("_")


def get_metrics_lists():
    return {
        name: list(values)
        for name, values in logs.items()
        if len(values) > 0
    }


def history_steps(history):
    lengths = [
        len(values)
        for values in history.values()
        if hasattr(values, "__len__")
    ]

    return max(lengths, default=0)


def atomic_torch_save(data, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    torch.save(data, temporary)
    os.replace(temporary, destination)


def atomic_json_save(data, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            default=str,
        )

    os.replace(temporary, destination)


def model_state_dict(model):
    # Avoid saving torch.compile's `_orig_mod.` prefix.
    state = model.state_dict()

    return {
        key.removeprefix("_orig_mod."): value
        for key, value in state.items()
    }


# ------------------------------------------------------------------
# Start
# ------------------------------------------------------------------

def start_run(name, config=None):
    global _active_run
    global current_run_name

    if _active_run is not None:
        raise RuntimeError(
            f"Run already active: {_active_run['run_id']}\n"
            "Call finish_run() before starting another."
        )

    orphan_steps = history_steps(get_metrics_lists())

    if orphan_steps > 0:
        raise RuntimeError(
            f"Found {orphan_steps:,} live log steps with no active run.\n"
            "Do not clear them yet. Save or inspect them first."
        )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H%M%S_%f"
    )

    short_name = sanitize_run_name(name)
    run_id = f"{timestamp}_{short_name}"

    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    _active_run = {
        "run_id": run_id,
        "name": short_name,
        "run_dir": run_dir,
        "started_at": datetime.now().isoformat(),
        "config": deepcopy(config or {}),
        "last_saved_steps": 0,
    }

    current_run_name = run_id
    logs.clear()

    metadata = {
        "run_id": run_id,
        "name": short_name,
        "status": "running",
        "started_at": _active_run["started_at"],
        "config": _active_run["config"],
    }

    atomic_json_save(
        metadata,
        run_dir / "metadata.json",
    )

    print(f"Started run:\n{run_id}")

    return run_id

def finish_run():
    global _active_run, current_run_name

    history = deepcopy(get_metrics_lists())
    steps = history_steps(history)
    finished_at = datetime.now().isoformat()

    if _active_run is None:
        if steps == 0:
            print("No active run or unsaved logs.")
            return

        # Preserve orphaned history separately. Current hyperparameters
        # and model weights may already belong to the next experiment.
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
        run_id = f"{timestamp}_recovered_logs"
        run_dir = RUN_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        record = {
            "run_id": run_id,
            "name": "recovered_logs",
            "run_dir": run_dir,
            "started_at": None,
            "config": {},
            "previous_run_hint": current_run_name,
            "status": "recovered",
        }
    else:
        record = deepcopy(_active_run)
        run_id = record["run_id"]
        run_dir = Path(record["run_dir"])
        record["status"] = "finished"

    record["finished_at"] = finished_at
    record["last_saved_steps"] = steps

    # Versioned filename avoids overwriting an earlier history file.
    save_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    history_path = run_dir / f"history_{save_stamp}.pt"

    atomic_torch_save(history, history_path)
    record["history_file"] = history_path.name

    metadata = {
        key: value
        for key, value in record.items()
        if key != "run_dir"
    }
    atomic_json_save(metadata, run_dir / "metadata.json")

    # Clear live state only after both saves succeed.
    runs[run_id] = {**record, "history": history}
    logs.clear()
    _active_run = None
    current_run_name = None

    print(f"Saved {steps:,} log steps:\n{history_path}")
    return run_id

# %% cell 34
def start_z_gradient_diagnostics(z, **weighted_regularizers):
    if not TRACK_Z_DIAGNOSTICS:
        return None

    z.retain_grad()

    gradients = {}
    for name, loss in weighted_regularizers.items():
        gradients[name] = torch.autograd.grad(
            loss,
            z,
            retain_graph=True,
        )[0].detach()

    # retain_grad() can capture gradients from the diagnostic passes.
    # Clear them so the real backward supplies only the total gradient.
    z.grad = None

    return gradients


def finish_z_gradient_diagnostics(enc, diagnostic):
    if diagnostic is None:
        return {}

    if z.grad is None:
        raise RuntimeError("Call finish diagnostics after ae_loss.backward().")

    with torch.no_grad():
        g_remaining = z.grad.detach().float().clone()

        for gradient in diagnostic.values():
            g_remaining.sub_(gradient.float())

        remaining_rms = g_remaining.square().mean().sqrt()

        metrics = {"remaining_grad_rms": remaining_rms}
        for name, gradient in diagnostic.items():
            rms = gradient.float().square().mean().sqrt()
            metrics[f"{name}_grad_rms"] = rms
            metrics[f"{name}_grad_ratio"] = rms / remaining_rms.clamp_min(1e-12)

    return metrics

# %% cell 35
def align_sigreg_grad(g_recon, g_sig, mode="plain", eps=1e-8):
    """
    Shape SIGReg's gradient relative to reconstruction.
    Projection is per image over the full latent [T, C].
    """
    shape = g_sig.shape

    gr = g_recon.flatten(1)
    gs = g_sig.flatten(1)

    # signed amount of SIGReg gradient parallel to recon gradient
    c = (
        (gs * gr).sum(dim=1, keepdim=True)
        / gr.square().sum(dim=1, keepdim=True).clamp_min(eps)
    )

    g_parallel = c * gr
    g_perp = gs - g_parallel

    if mode == "plain":
        out = gs

    elif mode == "sovereign":
        # SIGReg gets only directions locally neutral to recon
        out = g_perp

    elif mode == "protected":
        # remove only anti-recon parallel component
        out = gs - c.clamp(max=0) * gr

    elif mode == "reinforce":
        # keep only parallel component that helps recon
        out = c.clamp(min=0) * gr

    else:
        raise ValueError(f"unknown mode: {mode}")

    return out.view(shape)

# %% cell 36
def save_run_snapshot(epoch=None):
    global _active_run

    if _active_run is None:
        raise RuntimeError(
            "No active run. Call start_run() first."
        )

    history = deepcopy(get_metrics_lists())
    new_steps = history_steps(history)

    if new_steps == 0:
        raise RuntimeError(
            "Refusing to save an empty history."
        )

    run_dir = _active_run["run_dir"]

    history_path = run_dir / "history.pt"
    checkpoint_path = run_dir / "checkpoint.pt"

    # Prevent a cleared/short history from replacing
    # a longer history already on disk.
    if history_path.exists():
        existing = torch.load(
            history_path,
            map_location="cpu",
            weights_only=False,
        )

        old_steps = history_steps(
            existing.get("history", {})
        )

        if new_steps < old_steps:
            raise RuntimeError(
                f"Refusing to replace {old_steps:,} "
                f"saved steps with {new_steps:,} steps."
            )

    atomic_torch_save(
        {
            "run_id": _active_run["run_id"],
            "epoch": epoch,
            "steps_per_epoch": len(train_dataloader),
            "history": history,
        },
        history_path,
    )

    atomic_torch_save(
        {
            "run_id": _active_run["run_id"],
            "epoch": epoch,
            "steps": new_steps,
            "encoder": model_state_dict(encoder),
            "decoder": model_state_dict(decoder),
        },
        checkpoint_path,
    )

    _active_run["last_saved_steps"] = new_steps
    _active_run["last_epoch"] = epoch

    print(
        f"Saved epoch {epoch}: "
        f"{new_steps:,} steps"
    )

# %% cell 37
def capture_run_config():
    names = [
        "use_masked_view",
        "view_mode",
        "view_weight",
        "lr",
        "laplacian_weight",
        "swd_weight",
        "bwd_weight",
        "c_swd_weight",
        "t_swd_weight",
        "ae_emb_dim",
        "n_ae_latent_tokens",
        "emb_dim",
        "n_latent_tokens",
        "num_blocks",
        "n_heads",
        "head_size",
        "patch_size",
    ]

    config = {
        name: globals()[name]
        for name in names
        if name in globals()
    }

    config["batch_size"] = train_dataloader.batch_size

    return config

# %% cell 38
#training readout
import time
import torch


class TrainingReadout:
    def __init__(
        self,
        step_metrics,
        report_every=500,
        watch="test clean",
        trend_epochs=5,
    ):
        # Mapping: display label -> key returned by your training step.
        self.step_metrics = step_metrics
        self.report_every = report_every
        self.watch = watch
        self.trend_epochs = trend_epochs
        self.total_steps = 0
        self.history = []

    @staticmethod
    def _number(value):
        if torch.is_tensor(value):
            return value.detach().item()
        return float(value)

    @staticmethod
    def _gain(old, new):
        if old == 0:
            return "—"
        return f"{100 * (old - new) / abs(old):+.1f}%"

    def start(self, title, total_epochs, evaluation):
        self.total_epochs = total_epochs
        self.history = [
            {key: self._number(value) for key, value in evaluation.items()}
        ]
        self.total_steps = 0

        print(f"\n{title} | {total_epochs} epochs")
        print(
            "START | "
            + " | ".join(
                f"{key} {value:.5f}"
                for key, value in self.history[0].items()
            )
        )

    def begin_epoch(self, epoch, phase=""):
        self.epoch = epoch
        self.epoch_steps = 0
        self.epoch_samples = 0
        self.epoch_sums = {}

        self.interval_steps = 0
        self.interval_samples = 0
        self.interval_sums = {}

        print(f"\nE{epoch}/{self.total_epochs} {phase}".rstrip())
        print(
            f"{'STEP':>8}"
            + "".join(f" {label:>10}" for label in self.step_metrics)
            + f" {'ms/step':>9} {'img/s':>8}"
        )

        self.epoch_start = time.perf_counter()
        self.interval_start = self.epoch_start

    def update(self, metrics, batch_size, last_batch=False):
        self.total_steps += 1
        self.epoch_steps += 1
        self.interval_steps += 1
        self.epoch_samples += batch_size
        self.interval_samples += batch_size

        for key in self.step_metrics.values():
            value = metrics[key]
            if torch.is_tensor(value):
                value = value.detach()

            # Keep tensor sums on-device; convert only when reporting.
            contribution = value * batch_size
            self.epoch_sums[key] = self.epoch_sums.get(key, 0) + contribution
            self.interval_sums[key] = (
                self.interval_sums.get(key, 0) + contribution
            )

        if self.interval_steps >= self.report_every or last_batch:
            averages = {
                key: self._number(value / self.interval_samples)
                for key, value in self.interval_sums.items()
            }

            # Read scalar values before measuring elapsed time so GPU work
            # needed for these metrics has finished.
            now = time.perf_counter()
            elapsed = max(now - self.interval_start, 1e-9)
            self.train_seconds = now - self.epoch_start

            print(
                f"{self.total_steps:8d}"
                + "".join(
                    f" {averages[key]:10.5f}"
                    for key in self.step_metrics.values()
                )
                + f" {1000 * elapsed / self.interval_steps:9.0f}"
                + f" {self.interval_samples / elapsed:8.0f}"
            )

            self.interval_steps = 0
            self.interval_samples = 0
            self.interval_sums = {}
            self.interval_start = time.perf_counter()

    def end_epoch(self, evaluation):
        current = {
            key: self._number(value) for key, value in evaluation.items()
        }
        previous = self.history[-1]
        self.history.append(current)

        n = self.trend_epochs
        older = self.history[-n - 1] if len(self.history) > n else None

        best_index = min(
            range(len(self.history)),
            key=lambda i: self.history[i][self.watch],
        )
        best = self.history[best_index][self.watch]
        best_at = "start" if best_index == 0 else f"E{best_index}"

        value = current[self.watch]
        gain_1 = self._gain(previous[self.watch], value)
        gain_n = self._gain(older[self.watch], value) if older else "—"

        print(
            f"\nE{self.epoch}/{self.total_epochs}"
            f" | {self.watch} {value:.5f}"
            f" | gain {gain_1}/1ep {gain_n}/{n}ep"
            f" | best {best:.5f} @{best_at}"
            f" | {self.train_seconds / 60:.2f} min"
        )
        print(
            "Other eval | "
            + " | ".join(
                f"{key} {value:.5f}"
                for key, value in current.items()
                if key != self.watch
            )
        )

# %% cell 39
# --------------------Def Init Models (and optimizers) --------------------


def init_models():
    
    encoder = Encoder(
        patch_size,
        c,
        ae_emb_dim,
        emb_dim,
        n_ae_latent_tokens,
        n_latent_tokens,
        num_blocks,
        n_heads,
        head_size,
    )
    
    decoder = Decoder(
        patch_size,
        c,
        ae_emb_dim,
        emb_dim,
        n_ae_latent_tokens,
        n_latent_tokens,
        num_blocks,
        n_heads,
        head_size,
    )
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoder.to(device)
    decoder.to(device)
    
    
    # -------------------- Optimizers --------------------
    
    ae_parameters = (
        list(encoder.parameters())
        + list(decoder.parameters())
    )
    n_ae_parameters  = sum(p.numel() for p in ae_parameters)
    print(f"{n_ae_parameters:,} ae parameters")
    
    ae_optimizer = torch.optim.AdamW(
        ae_parameters,
        lr=lr,
        betas=(BETA1, BETA2),
        fused=True,
    )

    return (
        encoder,
        decoder,
        ae_optimizer,
        device,
    )

# %% cell 40
######################################################################################################################################
############################################################ init models #############################################################
######################################################################################################################################


SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
train_generator.manual_seed(DATA_SEED)

(
    encoder,
    decoder,
    ae_optimizer,
    device,
) = init_models()
torch._functorch.config.donated_buffer = not TRACK_Z_DIAGNOSTICS

# %% cell 41
finish_run()

start_run(
    RUN_NAME,
    config=capture_run_config(),
)

# %% cell 42
render_reconstruction(); #test model recon

# %% cell 43
################## Training Step Methods ########################

def make_pixel_mask(mask, images):
    B, _, H, W = images.shape

    return (
        mask.reshape(B, 1, pr, pc)
        .repeat_interleave(H // pr, dim=2)
        .repeat_interleave(W // pc, dim=3)
        .to(images.dtype)
    )


def masked_mse(prediction, target, pixel_mask):
    errors = F.mse_loss(
        prediction,
        target,
        reduction="none",
    )

    missing_values = (
        pixel_mask.sum() * target.shape[1]
    ).clamp_min(1)

    return (errors * pixel_mask).sum() / missing_values


def make_mask(images, mask_ratio=0.5, return_keep_indices=False):
    B = images.shape[0]
    T = pr * pc
    n_mask = int(mask_ratio * T)

    mask_indices = torch.rand(
        B, T, device=images.device,
    ).topk(
        k=n_mask,
        dim=1,
        largest=False,
    ).indices

    mask = torch.zeros(
        B, T,
        dtype=torch.bool,
        device=images.device,
    )

    mask.scatter_(dim=1, index=mask_indices, value=True)

    if return_keep_indices:
        n_keep = T - n_mask

        # False (visible) sorts before True (masked).
        # Stable sorting keeps visible patches in original image order.
        keep_indices = mask.int().argsort(
            dim=1, stable=True,
        )[:, :n_keep]

        return mask, keep_indices

    return mask


def make_train_step(
    use_masked_view=True,
    view_mode="symmetric",
):
    """
    use_masked_view:
        False -> pure reconstruction
        True  -> clean + masked reconstruction

    view_mode:
        "none"
            No latent view-alignment loss.

        "symmetric"
            Both clean and masked latents receive gradients:
            MSE(z_masked, z_clean)

        "clean_target"
            Clean latent is the stop-grad target:
            MSE(z_masked, z_clean.detach())

    These settings are fixed when the train step is created,
    which keeps the compiled graph static.
    """

    if view_mode not in {
        "none",
        "symmetric",
        "clean_target",
    }:
        raise ValueError(
            f"Unknown view_mode: {view_mode}"
        )

    if not use_masked_view and view_mode != "none":
        raise ValueError(
            "view_mode must be 'none' when use_masked_view=False"
        )

    def train_step(images, optimizer, view_weight):

        B = images.shape[0]

        optimizer.zero_grad(set_to_none=True)


        # --------------------------------
        # Masks
        # --------------------------------

        if use_masked_view:
            mask, keep_indices = make_mask(
                images,
                mask_ratio=0.5,
                return_keep_indices=True,
            )
            pixel_mask = make_pixel_mask(
                mask,
                images,
            )

        # --------------------------------
        # Forward
        # --------------------------------

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            z_clean = encoder(images)
        
            recon_clean = decoder(
                z_clean,
            )
        
            if use_masked_view:
                z_masked = encoder(
                    images,
                    keep_indices=keep_indices,
                )
        
                recon_masked = decoder(
                    z_masked,
                )
        
        z_clean = z_clean.float()
        recon_clean = recon_clean.float()
        
        if use_masked_view:
            z_masked = z_masked.float()
            recon_masked = recon_masked.float()

        # --------------------------------
        # Reconstruction
        # --------------------------------

        clean_recon = F.mse_loss(
            recon_clean,
            images,
        )

        if use_masked_view:
            masked_recon = masked_mse(
                recon_masked,
                images,
                pixel_mask,
            )

            clean_term = 0.5 * clean_recon
            masked_term = 0.5 * masked_recon

        else:
            clean_term = clean_recon
            masked_term = z_clean.new_zeros(())

        # --------------------------------
        # View alignment
        # --------------------------------

        if view_mode == "symmetric":
            view_term = view_weight * F.mse_loss(
                z_masked,
                z_clean,
            )

        elif view_mode == "clean_target":
            view_term = view_weight * F.mse_loss(
                z_masked,
                z_clean.detach(),
            )

        else:
            view_term = z_clean.new_zeros(())

        # --------------------------------
        # Laplacian
        # --------------------------------

        image_highfreq = laplacian(images)

        clean_laplacian = F.mse_loss(
            laplacian(recon_clean),
            image_highfreq,
        )

        if use_masked_view:
            masked_laplacian = masked_mse(
                laplacian(recon_masked),
                image_highfreq,
                pixel_mask,
            )

            laplacian_term = (
                0.5
                * laplacian_weight
                * (
                    clean_laplacian
                    + masked_laplacian
                )
            )

        else:
            masked_laplacian = z_clean.new_zeros(())

            laplacian_term = (
                laplacian_weight
                * clean_laplacian
            )

        # --------------------------------
        # Latent regularization
        # --------------------------------

        prior = torch.randn_like(z_clean)

        clean_swd = sliced_wasserstein_distance(
            z_clean,
            prior,
        )

        clean_bwd = basis_wasserstein_distance(
            z_clean,
            prior,
        )

        clean_c_swd = channel_sliced_wasserstein_distance(
            z_clean,
            prior,
        )

        clean_t_swd = token_sliced_wasserstein_distance(
            z_clean,
            prior,
        )

        if use_masked_view:
            masked_swd = sliced_wasserstein_distance(
                z_masked,
                prior,
            )

            masked_bwd = basis_wasserstein_distance(
                z_masked,
                prior,
            )

            masked_c_swd = channel_sliced_wasserstein_distance(
                z_masked,
                prior,
            )

            masked_t_swd = token_sliced_wasserstein_distance(
                z_masked,
                prior,
            )

            swd_term = (
                swd_weight
                * 0.5
                * (clean_swd + masked_swd)
            )

            bwd_term = (
                bwd_weight
                * 0.5
                * (clean_bwd + masked_bwd)
            )

            c_swd_term = (
                c_swd_weight
                * 0.5
                * (clean_c_swd + masked_c_swd)
            )

            t_swd_term = (
                t_swd_weight
                * 0.5
                * (clean_t_swd + masked_t_swd)
            )

        else:
            masked_swd = z_clean.new_zeros(())
            masked_bwd = z_clean.new_zeros(())
            masked_c_swd = z_clean.new_zeros(())
            masked_t_swd = z_clean.new_zeros(())

            swd_term = (
                swd_weight * clean_swd
            )

            bwd_term = (
                bwd_weight * clean_bwd
            )

            c_swd_term = (
                c_swd_weight * clean_c_swd
            )

            t_swd_term = (
                t_swd_weight * clean_t_swd
            )

        # --------------------------------
        # Total
        # --------------------------------

        ae_loss = (
            clean_term
            + masked_term
            + laplacian_term
            + view_term
            + swd_term
            + bwd_term
            + c_swd_term
            + t_swd_term
        )

        ae_loss.backward()
        # (skipped for the LR guard's ScoreOnly probe stub, which has no params)
        if GRAD_CLIP > 0 and hasattr(optimizer, "param_groups"):
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], GRAD_CLIP
            )
        optimizer.step()

        # --------------------------------
        # Metrics
        # --------------------------------

        return {
            "ae_loss": ae_loss.detach(),

            "clean_term": clean_term.detach(),
            "masked_term": masked_term.detach(),
            "laplacian_term": laplacian_term.detach(),
            "view_term": view_term.detach(),

            "swd_term": swd_term.detach(),
            "bwd_term": bwd_term.detach(),
            "c_swd_term": c_swd_term.detach(),
            "t_swd_term": t_swd_term.detach(),

            "clean_swd": clean_swd.detach(),
            "masked_swd": masked_swd.detach(),

            "clean_bwd": clean_bwd.detach(),
            "masked_bwd": masked_bwd.detach(),

            "clean_c_swd": clean_c_swd.detach(),
            "masked_c_swd": masked_c_swd.detach(),

            "clean_t_swd": clean_t_swd.detach(),
            "masked_t_swd": masked_t_swd.detach(),
        }

    return train_step


def train_step_refine_decoder(
    images,
    optimizer,
    noise_sigma,
):
    B = images.shape[0]

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        with torch.no_grad():
            z = encoder(images)

            noise_mask = (
                torch.rand(
                    (B, 1, 1),
                    device=images.device,
                ) < 0.5
            )

            z = z + (
                torch.randn_like(z)
                * noise_sigma
                * noise_mask
            )

        x_hat = decoder(z)

    x_hat = x_hat.float()

    recon_loss = F.mse_loss(
        x_hat,
        images,
    )

    recon_loss.backward()
    optimizer.step()

    return {
        "recon": recon_loss.detach(),
    }

# %% cell 44
#make and compile the training steps
train_step = make_train_step(
    use_masked_view=use_masked_view,
    view_mode=view_mode,
)

# COMPILE_MODE (overnight loop): torch.compile mode; empty/"default" = notebook
# behaviour (mode=None). e.g. max-autotune-no-cudagraphs, reduce-overhead.
COMPILE_MODE = _env("COMPILE_MODE", "", str)
train_step = torch.compile(
    train_step,
    mode=(COMPILE_MODE or None) if COMPILE_MODE != "default" else None,
)

# %% cell 45
# fixed_micro_batch_size = 6
# fixed_micro_batch = next(iter(train_dataloader))[0][:fixed_micro_batch_size].to(device)
# fixed_mask = make_mask(fixed_micro_batch)

# %% cell 46
# #OVERFIT MICRO-BATCH
# num_epochs = 10


# encoder.train()
# decoder.train()

# report_every = 500

# report_start = time.perf_counter()

# train_loss = get_split_loss('train')
# test_loss = get_split_loss('test')

# print(f'START TRAINING | recon-train-loss:{train_loss:.5f}  recon-test-loss: {test_loss:.5f}')

# for epoch in range(num_epochs): 
#     for batch_step, (images, _) in enumerate(train_dataloader):

#         total_steps = epoch*batches_per_epoch + batch_step

#         metrics = train_step_simple(fixed_micro_batch)

#         if ((total_steps+1) % report_every) == 0:

#             elapsed = time.perf_counter() - report_start
#             sec_per_step = elapsed / report_every
#             samples_per_sec = images.shape[0] / sec_per_step

#             report_start = time.perf_counter()

#             print(
#                 f"\nstep {total_steps + 1} | "
#                 f"{sec_per_step * 1000:.0f} ms/step | "
#                 f"{samples_per_sec:.1f} samples/s | "
#                 f"{len(train_dataloader) * sec_per_step / 60:.1f} min/epoch "

#                 f" --- GPU alloc={torch.cuda.memory_allocated()/1e9:.2f} GB | "
#                 f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB | "
#                 f"peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB"
#             )
            
#             msg = (
#                 f"AE:"
#                 f" total={metrics['ae_loss']:.5f}"
#                 f" | recon={metrics['recon_loss']:.5f}"
#                 f" | swd={metrics['swd_loss']:.5f}"
#                 f" | bwd={metrics['bwd_loss']:.5f}"
#                 f" | laplacian={metrics['laplacian_loss']:.5f}"
#             )
            
#             if TRACK_Z_DIAGNOSTICS:
#                 msg += (
#                     f" | s/r-ratio={metrics['grad_ratio']:.5f}"
#                     f" | v/r-ratio={metrics['variance_ratio']:.5f}"
#                 )
                
#             print(msg)


#     #render_reconstruction()
#     train_loss = get_split_loss('train')
#     test_loss = get_split_loss('test')
#     print(' ')
#     print('--------------------------------------------------------------')
#     print(f'EPOCH:{epoch} | recon-train-loss:{train_loss:.5f}  recon-test-loss: {test_loss:.5f}')
#     print('--------------------------------------------------------------')

# %% cell 47
# for image in fixed_micro_batch:
#     reconstruct_images(image[None, ...])

# %% cell 48
from pathlib import Path

preview_dir = Path("training_runs") / current_run_name / "latest"
preview_dir.mkdir(parents=True, exist_ok=True)

print(preview_dir)

# %% cell 49
#setup cell: shrink only spike guard
from collections import deque
from copy import deepcopy
import math

# Eager copy of your exact training objective.
_probe_step = make_train_step(
    use_masked_view=use_masked_view,
    view_mode=view_mode,
)

lr_events = []
_lr_guard = {
    "step": 0,
    "fast": None,
    "slow": None,
    "last_probe": -500,
}
_probe_batches = deque(maxlen=4)
# overnight loop knob: a probe scale must beat the current lr's loss by this
# factor before lr is reduced (notebook default 0.995 = 0.5% margin).
GUARD_GAIN = float(os.environ.get("GUARD_GAIN", 0.995))
# overnight loop knob: GUARD_PROBES=0 keeps the guard's loss EMA / non-finite
# check but never fires an LR probe (and retains no probe batches).
# Default 1 = notebook behaviour.
GUARD_PROBES = bool(int(os.environ.get("GUARD_PROBES", 1)))
# overnight loop knob: SYNC_EVERY=N reads the loss to the host (guard EMA /
# non-finite check, per-step metric logging) only every N steps so the CPU
# can queue kernels ahead of the GPU. Default 1 = notebook (sync every step).
SYNC_EVERY = max(1, int(os.environ.get("SYNC_EVERY", 1)))


def _probe_smaller_lr(optimizer, view_weight):
    parameters = [
        p for group in optimizer.param_groups
        for p in group["params"]
    ]
    weights = [p.detach().clone() for p in parameters]
    gradients = [
        None if p.grad is None else p.grad.detach().clone()
        for p in parameters
    ]
    buffers = [
        (b, b.detach().clone())
        for model in (encoder, decoder)
        for b in model.buffers()
    ]
    opt_state = deepcopy(optimizer.state_dict())
    cpu_rng = torch.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else None
    )

    scales = [0.125, 0.25, 0.5, 0.75, 1.0]
    scores = []

    def restore_rng(cpu, cuda):
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)

    @torch.no_grad()
    def restore_model():
        for p, saved in zip(parameters, weights):
            p.copy_(saved)
        for b, saved in buffers:
            b.copy_(saved)

    # Reuse your train step for scoring, but don't update Adam
    # or weights. This still runs backward, so it isn't optimized
    # for speed, but needs no rewrite of your loss function.
    class ScoreOnly:
        def zero_grad(self, set_to_none=True):
            optimizer.zero_grad(set_to_none=set_to_none)

        def step(self):
            pass

    try:
        for images in _probe_batches:
            restore_model()
            optimizer.load_state_dict(deepcopy(opt_state))

            batch_cpu = torch.get_rng_state()
            batch_cuda = (
                torch.cuda.get_rng_state_all()
                if cuda_rng is not None else None
            )

            # One actual AdamW proposal from the saved state.
            baseline = float(_probe_step(
                images, optimizer, view_weight
            )["ae_loss"].item())

            if not math.isfinite(baseline) or baseline <= 0:
                raise RuntimeError("Invalid baseline loss in LR probe.")

            delta = [
                p.detach().clone() - saved
                for p, saved in zip(parameters, weights)
            ]

            row = []
            for scale in scales:
                restore_model()
                with torch.no_grad():
                    for p, saved, change in zip(
                        parameters, weights, delta
                    ):
                        p.copy_(saved + scale * change)

                # Identical stochastic draws within this sweep.
                restore_rng(batch_cpu, batch_cuda)
                loss = float(_probe_step(
                    images, ScoreOnly(), view_weight
                )["ae_loss"].item())

                row.append(
                    loss / baseline
                    if math.isfinite(loss) else float("inf")
                )

            scores.append(row)

    finally:
        restore_model()
        optimizer.load_state_dict(deepcopy(opt_state))
        for p, grad in zip(parameters, gradients):
            p.grad = grad
        restore_rng(cpu_rng, cuda_rng)

    means = [
        sum(row[i] for row in scores) / len(scores)
        for i in range(len(scales))
    ]
    best = min(range(len(scales)), key=lambda i: means[i])

    # Require a meaningful gain over the current scale.
    # No zero option, and no increases in this first version.
    chosen = (
        scales[best]
        if math.isfinite(means[best])
        and means[best] < GUARD_GAIN * means[-1]
        else 1.0
    )
    chosen = max(0.5, chosen)
    return chosen, dict(zip(scales, means))


def train_step_with_lr_guard(images, optimizer, view_weight):
    # Exactly one normal training step; return its original metrics.
    metrics = train_step(
        images, optimizer=optimizer, view_weight=view_weight
    )

    state = _lr_guard
    state["step"] += 1
    if state["step"] % SYNC_EVERY:
        # No host sync this step (SYNC_EVERY > 1); EMA/guard skip it.
        return metrics
    loss = float(metrics["ae_loss"].item())

    if not math.isfinite(loss):
        raise RuntimeError(
            "Non-finite training loss. Stop and reload a known-good "
            "checkpoint; this guard cannot undo a corrupted update."
        )

    # EMA spans of 100 and 10,000 steps.
    for key, span in (("fast", 100), ("slow", 10000)):
        previous = state[key]
        state[key] = (
            loss if previous is None
            else previous + (2.0 / (span + 1)) * (loss - previous)
        )

    if GUARD_PROBES:
        _probe_batches.append(images.detach())

    ratio = state["fast"] / max(state["slow"], 1e-12)
    epoch_end = state["step"] % len(train_dataloader) == 0
    spike = ratio > 5.0

    # At most one spike-triggered probe per 500 steps.
    fire = (
        GUARD_PROBES
        and state["step"] >= 5000
        and (
            epoch_end
            or (spike and state["step"] - state["last_probe"] >= 500)
        )
    )

    if fire:
        reason = "epoch+spike" if epoch_end and spike else (
            "epoch" if epoch_end else "spike"
        )
        print(
            f"[LR probe] step={state['step']:,} "
            f"trigger={reason} EMA ratio={ratio:.2f}",
            flush=True,
        )

        old_lrs = [g["lr"] for g in optimizer.param_groups]
        scale, scores = _probe_smaller_lr(optimizer, view_weight)

        for group, old_lr in zip(optimizer.param_groups, old_lrs):
            group["lr"] = old_lr * scale

        event = {
            "step": state["step"],
            "trigger": reason,
            "ema_ratio": ratio,
            "scale": scale,
            "old_lrs": old_lrs,
            "new_lrs": [g["lr"] for g in optimizer.param_groups],
            "mean_relative_losses": scores,
        }
        lr_events.append(event)
        state["last_probe"] = state["step"]

        print(
            f"[LR probe] scale={scale:g} | "
            f"LR {old_lrs[0]:.3g} -> {event['new_lrs'][0]:.3g} | "
            + " ".join(f"{a:g}×:{v:.4f}" for a, v in scores.items()),
            flush=True,
        )

    return metrics

# %% cell 50
## AE EXPERIMENT TRAINING LOOP ##

num_epochs = int(os.environ.get("EPOCHS", 300))
report_every = 500
eval_batches = 20  # None = full loaders

encoder.requires_grad_(True)
decoder.requires_grad_(True)

#SET TO FALSE FOR DENSE FINE TUNING
sparse_middle = bool(_env("SPARSE_MIDDLE", 1, int))  # env override for overnight loop
encoder.sparse_middle_transformer.sparse_middle = sparse_middle
decoder.sparse_middle_transformer.sparse_middle = sparse_middle

train_optimizer = ae_optimizer

# for group in train_optimizer.param_groups:
#     group["lr"] = train_lr

readout = TrainingReadout(
    step_metrics={
        "clean": "clean_term",
        "missing": "masked_term",
        "view": "view_term",
        "swd": "swd_term",
        "bwd": "bwd_term",
        "c_swd": "c_swd_term",
        "t_swd": "t_swd_term",
    },
    report_every=report_every,
    watch="test clean",
    trend_epochs=5,
)

readout.start(
    title=(
        f"{current_run_name} | "
        f"masked={use_masked_view} | "
        f"view={view_mode} | "
        f"view_weight={view_weight:g}"
    ),
    total_epochs=num_epochs,
    evaluation=evaluate_multiview(
        max_batches=eval_batches,
    ),
)

steps_per_epoch = len(train_dataloader)

train_start = time.perf_counter()
out_of_time = False

# Optional wall-clock LR schedule (overnight loop): LR_SCHED=cosine gives
# linear warm-up over WARMUP_STEPS steps, then cosine decay from `lr` to
# lr*LR_MIN_FRAC at TIME_BUDGET_SEC. Unset = notebook behaviour (constant lr,
# only the LR guard changes it). Guard reductions are kept as a multiplier.
LR_SCHED = str(_env("LR_SCHED", "none", str))
WARMUP_STEPS = int(_env("WARMUP_STEPS", 300, int))
LR_MIN_FRAC = float(_env("LR_MIN_FRAC", 0.0))
# Fraction of TIME_BUDGET_SEC at which the cosine decay starts (0 = right
# after warm-up, as before; 0.75 = constant lr for the first 75% of the
# budget, then cosine to lr*LR_MIN_FRAC over the last 25%).
DECAY_START_FRAC = float(_env("DECAY_START_FRAC", 0.0))
_sched_mult = 1.0
_sched_last_target = None


def _scheduled_lr(step, elapsed):
    if step < WARMUP_STEPS:
        f = (step + 1) / WARMUP_STEPS
    else:
        t = min(elapsed / TIME_BUDGET_SEC, 1.0)
        if DECAY_START_FRAC >= 1.0:
            return lr  # warm-up only: constant lr after WARMUP_STEPS
        if DECAY_START_FRAC > 0:
            t = max(0.0, t - DECAY_START_FRAC) / (1.0 - DECAY_START_FRAC)
        f = LR_MIN_FRAC + (1.0 - LR_MIN_FRAC) * 0.5 * (1.0 + math.cos(math.pi * t))
    return lr * f


def _apply_lr_schedule(step, elapsed):
    global _sched_mult, _sched_last_target
    if LR_SCHED == "none":
        return
    groups = train_optimizer.param_groups
    if _sched_last_target is not None and groups[0]["lr"] != _sched_last_target:
        _sched_mult *= groups[0]["lr"] / _sched_last_target  # keep guard scaling
    target = _scheduled_lr(step, elapsed) * _sched_mult
    for g in groups:
        g["lr"] = target
    _sched_last_target = target


_global_step = 0

# PROFILE=1 (overnight loop): per-step split of dataloader wait / GPU step
# (with cuda sync) / post-step bookkeeping, epoch-end overhead, and a
# synthetic same-batch loop after the budget to isolate pure step cost.
PROFILE = bool(_env("PROFILE", 0, int))
# EVAL_EVERY / SNAPSHOT_EVERY (overnight loop): run the epoch-end eval only
# every N epochs (0 = final only) and the snapshots+checkpoint only every N
# epochs (0 = final only). Default 1/1 = notebook behaviour. The final
# (out-of-time) epoch always evaluates and saves.
EVAL_EVERY = _env("EVAL_EVERY", 1, int)
SNAPSHOT_EVERY = _env("SNAPSHOT_EVERY", 1, int)
_prof ={"data": 0.0, "step": 0.0, "post": 0.0, "n": 0}
_t_loop_end = time.perf_counter()

for epoch in range(num_epochs):
    encoder.train()
    decoder.train()

    readout.begin_epoch(
        epoch + 1,
        phase="TRAIN",
    )

    for batch_step, (images, _) in enumerate(train_dataloader):
        if PROFILE:
            _t0 = time.perf_counter()
            _prof["data"] += _t0 - _t_loop_end
        images = images.to(
            device,
            non_blocking=True,
        )

        _apply_lr_schedule(_global_step, time.perf_counter() - train_start)
        _global_step += 1

        metrics = train_step_with_lr_guard(
            images,
            optimizer=train_optimizer,
            view_weight=view_weight,
        )
        if PROFILE:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()
            _prof["step"] += _t1 - _t0

        if _global_step % SYNC_EVERY == 0:
            metric_values = collect_metrics(metrics)

            log_metrics(**metric_values)

        out_of_time = (
            time.perf_counter() - train_start >= TIME_BUDGET_SEC
        )

        readout.update(
            metrics,
            batch_size=images.shape[0],
            last_batch=(batch_step + 1 == steps_per_epoch) or out_of_time,
        )
        if PROFILE:
            _t_loop_end = time.perf_counter()
            _prof["post"] += _t_loop_end - _t1
            _prof["n"] += 1
            if _prof["n"] % report_every == 0:
                n = _prof["n"]
                print(
                    f"[profile] step {readout.total_steps:,} | per step ms: "
                    f"data {1000 * _prof['data'] / n:.1f} | "
                    f"gpu-step {1000 * _prof['step'] / n:.1f} | "
                    f"post {1000 * _prof['post'] / n:.1f}",
                    flush=True,
                )
                _prof.update(data=0.0, step=0.0, post=0.0, n=0)

        if out_of_time:
            break

    #cell display
    _do_eval = out_of_time or (EVAL_EVERY > 0 and (epoch + 1) % EVAL_EVERY == 0)
    _do_snap = out_of_time or (SNAPSHOT_EVERY > 0 and (epoch + 1) % SNAPSHOT_EVERY == 0)
    _t_eval = time.perf_counter()
    if _do_eval:
        final_eval = evaluate_multiview(
            max_batches=eval_batches,
        )
        readout.end_epoch(final_eval)
    else:
        print(
            f"\nE{epoch + 1}/{num_epochs} | eval skipped (EVAL_EVERY={EVAL_EVERY}) | "
            f"step {readout.total_steps:,} | "
            f"{(time.perf_counter() - train_start) / 60:.2f} min",
            flush=True,
        )
    if PROFILE:
        print(f"[profile] epoch-end eval {time.perf_counter() - _t_eval:.2f}s", flush=True)

    if out_of_time:
        print(
            f"\nTIME BUDGET {TIME_BUDGET_SEC:.0f}s reached | "
            f"step {readout.total_steps:,} | "
            f"{(time.perf_counter() - train_start) / 60:.2f} min | "
            f"FINAL train clean {final_eval['train clean']:.5f} | "
            f"test clean {final_eval['test clean']:.5f}",
            flush=True,
        )
        if PROFILE:
            # Synthetic loop on the last batch (after the final eval, so it
            # does not affect reported numbers): step cost without dataloader.
            for label, sync_each in (
                ("train_step + .item() sync", True),
                ("train_step, no per-step sync", False),
            ):
                torch.cuda.synchronize()
                _ts = time.perf_counter()
                for _ in range(200):
                    m = train_step(images, optimizer=train_optimizer, view_weight=view_weight)
                    if sync_each:
                        float(m["ae_loss"].item())
                torch.cuda.synchronize()
                print(
                    f"[profile] synthetic 200 steps, {label}: "
                    f"{1000 * (time.perf_counter() - _ts) / 200:.1f} ms/step",
                    flush=True,
                )
            if bool(_env("PROFILE_TRACE", 0, int)):
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof_trace:
                    for _ in range(20):
                        train_step(images, optimizer=train_optimizer, view_weight=view_weight)
                    torch.cuda.synchronize()
                print("[profile] kernel table (20 synthetic steps), by CUDA time:", flush=True)
                print(prof_trace.key_averages().table(sort_by="cuda_time_total", row_limit=30), flush=True)

    #saving/logging
    _t_snap = time.perf_counter()
    if _do_snap:
        save_loss_snapshots()
        save_reconstruction_snapshot(masked=False)
        save_reconstruction_snapshot(masked=True)

        completed_epoch = epoch + 1

        save_run_snapshot(
            epoch=completed_epoch,
        )

        atomic_torch_save(
            lr_events,
            _active_run["run_dir"] / "lr_events.pt",
        )

        plt.close("all")
    if PROFILE:
        print(f"[profile] epoch-end snapshots+checkpoint {time.perf_counter() - _t_snap:.2f}s", flush=True)
        _t_loop_end = time.perf_counter()

    if out_of_time:
        break

