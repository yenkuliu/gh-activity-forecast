"""Command-line training, cohort prediction and historical evaluation."""
import argparse
import csv
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .features import EVENTS, GAP, WEEK, FeatureBuilder, feature_names, target_seconds
from .feature_tables import PRIMARY_KEYS
from .protocol import labeled_examples, weekly_split


def table_hashes(folder):
    checksums = {}
    for name in PRIMARY_KEYS:
        path = folder / (name + ".parquet")
        h = sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        checksums[path.name] = h.hexdigest()
    return checksums


def train(args):
    from .training import fit_ensemble

    protocol = weekly_split(args.target)
    if args.out.exists():
        raise FileExistsError(f"Model directory already exists: {args.out}")
    builder = FeatureBuilder(args.tables)
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = {"split": protocol, "seed": args.seed, "activities": list(EVENTS),
                "feature_names": {k: list(feature_names(k)) for k in ("compact", "enriched")},
                "table_sha256": table_hashes(args.tables),
                "packages": {k: version(k) for k in ("torch", "numpy", "pandas", "pyarrow", "scipy")}}
    (args.out / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    training = labeled_examples(builder, protocol["training"], protocol["training_label_origin"])
    selection = labeled_examples(builder, protocol["epoch_selection"], protocol["selection_label_origin"])
    calibration = labeled_examples(builder, [protocol["threshold_calibration"]], protocol["calibration_label_origin"])
    cohort_seen = builder.db.user_seen[builder.db.cohort]
    calibration["inference_prefix_rows"] = sum(
        int(np.count_nonzero(cohort_seen < target_seconds(w) - GAP))
        for w in protocol["calibration_candidates"][:-1]
    )
    fit_ensemble(training, selection, calibration, args.out, seed=args.seed, device=args.device)
    print(f"Saved the trained ensemble to {args.out}")


def read_users(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = csv.reader(f)
        if next(rows, []) != ["UserID"]:
            raise ValueError("The input CSV must contain exactly one column named UserID")
        users = []
        for row in rows:
            if len(row) != 1 or not row[0].isascii() or not row[0].isdecimal():
                raise ValueError(f"Invalid UserID on line {rows.line_num}")
            users.append(int(row[0]))
    if not users or len(set(users)) != len(users):
        raise ValueError("The input CSV must have at least one user and unique IDs")
    return users


def write_csv(path, ids, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    if values.shape != (len(ids), len(EVENTS)) or not np.isin(values, [0, 1]).all():
        raise ValueError("Expected six binary values for each user")
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, newline="", encoding="utf-8",
                                     delete=False) as f:
        temp = Path(f.name)
        try:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["UserID", *EVENTS])
            for user, row in zip(ids, values):
                writer.writerow([int(user), *map(int, row)])
            f.flush()
            os.fsync(f.fileno())
            f.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def apply_model(args):
    from .training import predict_ensemble

    metadata = json.loads((args.model / "dataset.json").read_text(encoding="utf-8"))
    fitted_for = metadata["split"]["target"]
    date = args.target or fitted_for
    target = target_seconds(date)
    if target < target_seconds(fitted_for):
        raise ValueError(f"The model was fitted for {fitted_for}; choose that date or a later one")
    historical = args.command == "backtest"
    if historical:
        end = datetime.fromisoformat(args.observed_through)
        if end.tzinfo is None or end.timestamp() < target + WEEK:
            raise ValueError("Provide an explicit time zone and cover the complete seven-day label horizon")
        if args.out.exists():
            raise FileExistsError(f"Evaluation directory already exists: {args.out}")
    else:
        if args.output.resolve() == args.users.resolve():
            raise ValueError("The forecast output must be separate from the input user list")
        users = read_users(args.users)
    if metadata["activities"] != list(EVENTS) or any(
        metadata["feature_names"][k] != list(feature_names(k)) for k in ("compact", "enriched")
    ):
        raise ValueError("Saved model and inference feature schemas differ")

    builder = FeatureBuilder(args.tables)
    sample = builder.snapshot(target, labels=historical)
    ids = sample["ids"]
    if not len(ids) or np.unique(ids).size != len(ids):
        raise ValueError("Expected unique, cutoff-visible cohort users")
    if not historical:
        index = {int(user): i for i, user in enumerate(ids)}
        missing = [u for u in users if u not in index]
        if missing:
            raise ValueError(f"{len(missing)} requested users missing from the visible cohort; check the tables and target date")
    probabilities, predictions = predict_ensemble(sample, args.model, device=args.device)
    if historical:
        truth = (sample["positive_seen"] < int(end.timestamp())).astype(np.uint8)
        args.out.mkdir(parents=True, exist_ok=False)
        write_csv(args.out / "predictions.csv", ids, predictions)
        write_csv(args.out / "truth.csv", ids, truth)
        np.savez_compressed(args.out / "probabilities.npz", ids=ids, probabilities=probabilities)
        print(f"Wrote predictions and truth for {len(ids)} cohort users to {args.out}")
    else:
        write_csv(args.output, users, predictions[[index[u] for u in users]])
        print(f"Wrote {len(users)} forecasts to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train and calibrate all three TabM members")
    prediction = commands.add_parser("predict", help="Export six binary activities per requested UserID")
    backtest = commands.add_parser("backtest", help="Score a historical week using the complete visible cohort")
    for command in (training, prediction, backtest):
        command.add_argument("--tables", required=True, type=Path, help="Directory with the eight official Parquet files")
        command.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
        command.add_argument("--target", required=command is training, help="Target week start YYYY-MM-DD (Taiwan); defaults to the model target at inference")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--out", type=Path, required=True, help="New model directory")
    for command in (prediction, backtest):
        command.add_argument("--model", type=Path, required=True)
    prediction.add_argument("--users", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    backtest.add_argument("--observed-through", required=True, help="Exclusive label observation end, ISO timestamp with UTC offset")
    backtest.add_argument("--out", type=Path, required=True, help="New directory for historical predictions and truth")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        apply_model(args)


if __name__ == "__main__":
    main()
