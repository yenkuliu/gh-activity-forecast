"""Weekly time splits and label observability for a Sunday forecast in Taiwan."""
from datetime import date, timedelta

import numpy as np

from .features import DAY, GAP, WEEK, target_seconds

FIRST_WEEK = date(2026, 6, 7)


def weekly_split(target):
    end = date.fromisoformat(target)
    if end.weekday() != 6:
        raise ValueError("The target must be a Sunday (Taiwan local date)")
    cutoff = target_seconds(target) - GAP
    weeks = []
    week = FIRST_WEEK
    while target_seconds(week.isoformat()) + WEEK <= cutoff:
        weeks.append(week.isoformat())
        week += timedelta(days=7)
    if len(weeks) < 4:
        raise ValueError("Four complete historical calibration weeks are required")
    candidates = weeks[-4:]
    fit_origin = target_seconds(candidates[0]) - GAP
    select_origin = target_seconds(candidates[-1]) - GAP
    training = [w for w in weeks if fit_origin - 90 * DAY <= target_seconds(w)
                and target_seconds(w) + WEEK <= fit_origin]
    if len(training) < 4:
        raise ValueError("The archive does not contain four complete training weeks")
    selection = [w for w in candidates if target_seconds(w) + WEEK <= select_origin]
    return {
        "target": target, "feature_cutoff_utc_seconds": cutoff,
        "training": training, "training_label_origin": fit_origin,
        "epoch_selection": selection, "selection_label_origin": select_origin,
        "calibration_candidates": candidates, "threshold_calibration": candidates[-1],
        "calibration_label_origin": cutoff,
    }


def labeled_examples(builder, weeks, origin):
    rows = []
    for week in weeks:
        print(f"Building features and historical labels for {week}", flush=True)
        sample = builder.snapshot(target_seconds(week))
        observed = sample.pop("positive_seen") < origin
        mask = np.ones(observed.shape, np.float32)
        mask[:, 5] = sample.pop("push_known_by") < origin
        rows.append({"compact": sample["compact"], "enriched": sample["enriched"],
                     "y": observed.astype(np.float32), "mask": mask})
    return {key: np.concatenate([r[key] for r in rows], axis=0) for key in rows[0]}
