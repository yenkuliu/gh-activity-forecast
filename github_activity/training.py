"""Fit the three TabM members, calibrate them, and reload them for inference."""

import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import torch

from .calibration import blend, fit_blend, fit_thresholds, masked_f1_score
from .model import MEMBERS, initialize, make_model, masked_bce, probabilities

BATCH_SIZE = 256
MAX_EPOCHS = 24
PATIENCE = 5
MIN_DELTA = 1e-5


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _standardize(x, center, scale):
    return np.clip((x - center) / scale, -10, 10).astype(np.float32)


def _check_features(sample):
    n = len(sample["compact"])
    for key, dims in (("compact", 162), ("enriched", 372)):
        values = sample[key]
        if values.shape != (n, dims) or values.dtype != np.float32:
            raise ValueError(f"{key} features must be a ({n}, {dims}) float32 matrix")
        if not np.isfinite(values).all():
            raise ValueError(f"{key} has non-finite features")
    return n


def _check_examples(sample):
    n = _check_features(sample)
    if n == 0:
        raise ValueError("A training or calibration split contains no users")
    for key in ("y", "mask"):
        values = sample[key]
        if values.shape != (n, 6) or not np.isin(values, [0, 1]).all():
            raise ValueError(f"{key} must be a ({n}, 6) binary matrix")


def _fit_member(member, train, selection, calibration, directory, seed, device):
    device = initialize(seed, device)
    raw = train[member.features]
    center = raw.mean(axis=0)
    scale = np.maximum(raw.std(axis=0), 0.1)
    x = _standardize(raw, center, scale)
    selection_x = _standardize(selection[member.features], center, scale)
    y = np.asarray(train["y"], dtype=np.float32)
    mask = np.asarray(train["mask"], dtype=np.float32)
    selection_y = np.asarray(selection["y"], dtype=np.float32)
    selection_mask = np.asarray(selection["mask"], dtype=np.float32)
    model = make_model(member.inputs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    directory.mkdir()
    np.savez(directory / "normalization.npz", center=center, scale=scale)
    best = -np.inf
    best_epoch = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(y))
        model.train()
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.from_numpy(x[rows]).to(device))
            loss = masked_bce(
                logits, torch.from_numpy(y[rows]).to(device), torch.from_numpy(mask[rows]).to(device)
            )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            # Weight by row count before averaging within the optimizer step.
            # The final short batch follows the same reduction as a full batch.
            (loss * len(rows) / len(rows)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(rows)
        p = probabilities(model, selection_x, device)
        q = np.clip(p, 1e-7, 1 - 1e-7)
        losses = -(selection_y * np.log(q) + (1 - selection_y) * np.log1p(-q))
        per_label = (losses * selection_mask).sum(axis=0) / np.maximum(selection_mask.sum(axis=0), 1)
        log_loss = float(per_label.mean())
        f1 = masked_f1_score(selection_y, p >= 0.25, selection_mask)
        score = f1 if member.selection == "f1" else -log_loss
        history.append(dict(epoch=epoch, train_loss=total_loss / len(order),
                            selection_log_loss=log_loss, selection_f1=f1))
        if score > best + MIN_DELTA:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), directory / "weights.pt")
        _write_json(directory / "epochs.json", history)
        print(f"{member.name} {epoch:02d}  train {history[-1]['train_loss']:.5f}  "
              f"selection log loss {log_loss:.5f}  F1 {f1:.5f}", flush=True)
        if epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(torch.load(directory / "weights.pt", map_location=device, weights_only=True))
    _write_json(directory / "fit.json", dict(input_width=member.inputs, feature_set=member.features,
                selected_by=member.selection, best_epoch=best_epoch, seed=seed))
    return probabilities(
        model, _standardize(calibration[member.features], center, scale), device,
        prefix_rows=calibration.get("inference_prefix_rows", 0),
    )


def fit_ensemble(train, selection, calibration, out, *, device="auto", seed=42):
    """Fit using three chronological splits. All labels are censored as of their split cutoff."""
    for sample in (train, selection, calibration):
        _check_examples(sample)
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / m.name).exists() for m in MEMBERS) or (directory / "decision.json").exists():
        raise FileExistsError("Choose a fresh model directory")
    runtime = initialize(seed, device)
    _write_json(directory / "training_config.json", dict(
        python=sys.version.split()[0], platform=platform.system(), device=str(runtime),
        torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__, seed=seed,
        architecture="TabM", blocks=3, width=256, branches=16, dropout=0.15,
        optimizer="AdamW", learning_rate=0.001, weight_decay=0.01, gradient_clip=1.0,
        batch_size=BATCH_SIZE, inference_batch_size=512, max_epochs=MAX_EPOCHS,
        patience=PATIENCE, min_delta=MIN_DELTA, selection_threshold=0.25,
        rows={k: len(s["y"]) for k, s in [("training", train), ("selection", selection), ("calibration", calibration)]},
    ))
    stack = np.stack([
        _fit_member(m, train, selection, calibration, directory / m.name, seed, runtime)
        for m in MEMBERS
    ])
    y, mask = calibration["y"], calibration["mask"]
    weights = fit_blend(stack, y, mask)
    pc = blend(stack, weights)
    thresholds = fit_thresholds(y, pc, mask)
    np.savez_compressed(directory / "calibration_predictions.npz", probabilities=pc, y=y, mask=mask)
    _write_json(directory / "decision.json", dict(
        members=[m.name for m in MEMBERS], blend_weights=weights.tolist(), thresholds=thresholds.tolist(),
    ))


def predict_ensemble(features, model_dir, *, device="auto"):
    """Return arrays (probabilities, six binary decisions) in the supplied user order."""
    _check_features(features)
    directory = Path(model_dir)
    decision = json.loads((directory / "decision.json").read_text(encoding="utf-8"))
    if decision["members"] != [m.name for m in MEMBERS]:
        raise ValueError("Unrecognized ensemble members or order")
    weights = np.asarray(decision["blend_weights"], dtype=float)
    thresholds = np.asarray(decision["thresholds"], dtype=float)
    if (weights.shape != (6, 3) or not np.all(weights >= 0) or
            not np.allclose(weights.sum(axis=1), 1) or thresholds.shape != (6,) or
            not np.all((thresholds >= 0) & (thresholds <= 1))):
        raise ValueError("Invalid blend weights or thresholds")
    device = initialize(42, device)
    output = []
    for member in MEMBERS:
        folder = directory / member.name
        with np.load(folder / "normalization.npz", allow_pickle=False) as z:
            x = _standardize(features[member.features], z["center"], z["scale"])
        model = make_model(member.inputs).to(device)
        model.load_state_dict(torch.load(folder / "weights.pt", map_location=device, weights_only=True))
        output.append(probabilities(model, x, device))
    p = blend(output, weights)
    return p, (p >= thresholds).astype(np.uint8)
