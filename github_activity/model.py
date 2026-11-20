"""TabM architecture and deterministic initialization for weekly activity forecasting."""

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .vendor.tabm import TabM


@dataclass(frozen=True)
class Member:
    name: str
    features: str
    inputs: int
    selection: str


MEMBERS = (
    Member("compact_log_loss", "compact", 162, "log_loss"),
    Member("enriched_log_loss", "enriched", 372, "log_loss"),
    Member("compact_f1", "compact", 162, "f1"),
)
WIDTH = 256
BRANCHES = 16
LABELS = 6


def initialize(seed, device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(device)
    if device.type not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device {device}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return device


def _reserve_linear_draws(dimensions):
    # Pin the original CPU initialization stream for each feature width. Each
    # linear projection reserves a weight matrix followed by a bias vector.
    for inputs, outputs in dimensions:
        torch.rand(outputs, inputs, device="cpu")
        torch.rand(outputs, device="cpu")


def make_model(inputs):
    _reserve_linear_draws(((inputs, WIDTH), (WIDTH, WIDTH), (WIDTH, LABELS)))
    model = TabM.make(
        n_num_features=inputs, d_out=LABELS, n_blocks=3,
        d_block=WIDTH, dropout=0.15, k=BRANCHES,
    )
    _reserve_linear_draws(((2 * WIDTH, WIDTH), (WIDTH, WIDTH), (WIDTH, LABELS)))
    return model


def masked_bce(logits, y, mask):
    y = y[:, None].expand_as(logits)
    weight = mask[:, None].expand_as(logits)
    terms = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (terms * weight).sum() / weight.sum().clamp_min(1e-12)


@torch.inference_mode()
def probabilities(model, x, device, *, prefix_rows=0):
    """Infer in 512-row batches, preserving reference shapes when scoring a suffix."""
    if prefix_rows < 0:
        raise ValueError("Batch prefix length must be nonnegative")
    padding = int(prefix_rows) % 512
    if padding:
        # TabM runs independently per user in eval mode. Padding restores the
        # batch partition of a final week scored after earlier weekly snapshots.
        x = np.concatenate((np.zeros((padding, x.shape[1]), dtype=x.dtype), x))
    model.eval()
    result = np.empty((len(x), LABELS), np.float32)
    for start in range(0, len(x), 512):
        batch = torch.from_numpy(np.ascontiguousarray(x[start:start + 512])).to(device)
        result[start:start + len(batch)] = model(batch).sigmoid().mean(dim=1).cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Non-finite activity probabilities")
    return result[padding:]
