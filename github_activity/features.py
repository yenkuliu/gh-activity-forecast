"""Build compact and enriched TabM inputs from cutoff-visible GitHub events."""
import numpy as np

from .feature_tables import ActivityTables, DAY, EVENTS, GAP, WEEK, target_seconds


RELATIONS = ("repo", "peer", "incoming")
WINDOWS = (7, 28, 56, 112)
HALF_LIVES = (7, 28)
BEHAVIOR_WINDOWS = (1, 3, 14, 21, 42, 84)
BEHAVIOR_SUMMARIES = (
    "active_weeks", "recent_week_fraction", "gap_mean", "gap_cv",
    "normalized_silence", "burst", "any_activity", "activity_diversity",
)


def _compact_blocks():
    blocks = [("self_static_count", 6), ("self_static_active", 6)]
    def relations(period):
        return [(f"rel_{group}_{period}", size) for group, size in zip(RELATIONS, (7, 7, 6))]
    blocks.extend(relations("static"))
    for window in WINDOWS:
        blocks.extend((f"time_{stat}_{window}", 6) for stat in ("count", "active", "coverage"))
    blocks.append(("time_recency", 6))
    for half in HALF_LIVES:
        blocks.append((f"time_decay_{half}", 6))
        blocks.extend(relations(f"time{half}"))
    return blocks


COMPACT_NAMES = tuple(f"{name}_{i}" for name, n in _compact_blocks() for i in range(n))
BEHAVIOR_NAMES = tuple(
    f"time_behavior_{kind}_{w}_{k}" for w in BEHAVIOR_WINDOWS
    for kind in ("count", "active", "rate") for k in range(6)
) + tuple(f"time_behavior_{kind}_{k}" for kind in BEHAVIOR_SUMMARIES for k in range(6))
CONTRASTS = tuple(
    (f"rel_{group}_{period}_{k}", f"{own}_{k}", f"rel_{group}_relative_{period}_{k}")
    for group in RELATIONS
    for period, own in (("static", "self_static_count"), ("time7", "time_count_7"), ("time28", "time_count_28"))
    for k in range(6)
)


def feature_names(view):
    """Return ordered feature names, as used for model fitting and scoring."""
    if view == "compact":
        return list(COMPACT_NAMES)
    if view == "enriched":
        return list(COMPACT_NAMES + BEHAVIOR_NAMES) + [name for _, _, name in CONTRASTS]
    raise ValueError(f"Unknown feature view: {view}")


def _behavior(daily, coverage):
    # The enriched representation uses daily log-counts quantized to float16.
    # Convert back to float32 counts before computing rates and inter-event gaps.
    sequence = np.log1p(daily).astype(np.float16)
    count = np.expm1(sequence.astype(np.float32))
    active = count > 0
    parts = []
    for window in BEHAVIOR_WINDOWS:
        total = count[:, -window:].sum(1)
        exposure = coverage[-window:].sum(0)
        parts.extend([np.log1p(total), active[:, -window:].mean(1),
                      np.log1p(total * 7 / np.maximum(exposure, 1))])
    weeks = active.reshape(len(active), 16, 7, 6).any(2)
    parts.extend([weeks.mean(1), weeks[:, -4:].mean(1)])

    day = np.arange(112, dtype=np.int16)[None, :, None]
    last_active = np.maximum.accumulate(np.where(active, day, -1), axis=1)
    supported = active[:, 1:] & (last_active[:, :-1] >= 0)
    gaps = np.where(supported, day[:, 1:] - last_active[:, :-1], 0).astype(np.float32)
    n = supported.sum(1)
    mean = np.divide(gaps.sum(1), n, out=np.full((len(count), 6), 112.0, np.float32), where=n > 0)
    variance = np.maximum((gaps * gaps).sum(1) / np.maximum(n, 1) - mean * mean, 0)
    silence = np.where(active.any(1), np.argmax(active[:, ::-1], axis=1), 112)
    parts.extend([
        np.log1p(mean),
        np.sqrt(variance) / np.maximum(mean, 1),
        np.log1p(silence / np.maximum(mean, 1)),
        np.log1p(count[:, -7:].sum(1)) - np.log1p(count[:, -28:-7].sum(1) / 3),
    ])
    active_days = active.any(2).mean(1)
    diversity = active[:, -28:].any(1).mean(1)
    parts.extend([np.repeat(active_days[:, None], 6, axis=1), np.repeat(diversity[:, None], 6, axis=1)])
    return np.concatenate(parts, 1).astype(np.float32)


class FeatureBuilder:
    """Prepare one set of tables and reuse it for all weekly snapshots."""

    def __init__(self, tables_path):
        self.db = ActivityTables(tables_path)

    def snapshot(self, target, *, labels=True):
        """Build features at target minus four hours, in visible cohort row order.

        Pass labels=False for inference. Returned ids are the anonymized
        github_user_id values used by the official submission interface.
        """
        target = int(target)
        cutoff = target - GAP
        db = self.db
        rows = db.cohort[db.user_seen[db.cohort] < cutoff]
        events = db.history(cutoff)
        counts = db.counts(events)
        blocks = [np.log1p(counts[rows]), (counts[rows] > 0).astype(np.float32)]
        blocks.extend(part[rows] for part in db.relations(events, counts, cutoff))

        row_map = np.full(db.n_users, -1, dtype=np.int64)
        row_map[rows] = np.arange(len(rows))
        local = row_map[events.u.to_numpy()]
        bins = 111 - ((cutoff - 1 - events.time.to_numpy()) // DAY)
        kinds = events.kind.to_numpy()
        available = local >= 0
        daily = np.zeros((len(rows), 112, 6), dtype=np.float32)
        np.add.at(daily, (local[available], bins[available], kinds[available]), 1)
        coverage = np.ones((112, 6), dtype=np.float32)
        coverage[:, 5] = np.bincount(bins[kinds == 5], minlength=112) > 0

        for window in WINDOWS:
            recent = daily[:, -window:]
            blocks.extend([
                np.log1p(recent.sum(1)),
                ((recent > 0).sum(1) / window).astype(np.float32),
                np.broadcast_to(coverage[-window:].mean(0), (len(rows), 6)),
            ])
        active = daily > 0
        latest = np.where(active.any(1), np.argmax(active[:, ::-1], axis=1), 112)
        blocks.append(np.log1p(latest.astype(np.float32)))
        age = np.arange(111, -1, -1)[None, :, None]
        for half in HALF_LIVES:
            weights = np.exp2(-age / half)
            blocks.append(np.log1p((daily * weights).sum(1)).astype(np.float32))
            recent_events = events[events.time >= cutoff - half * DAY]
            recent_counts = db.counts(recent_events)
            blocks.extend(part[rows] for part in db.relations(recent_events, recent_counts, cutoff))

        compact = np.concatenate(blocks, axis=1)
        behavior = np.empty((len(rows), len(BEHAVIOR_NAMES)), dtype=np.float32)
        for start in range(0, len(rows), 512):
            batch = slice(start, start + 512)
            behavior[batch] = _behavior(daily[batch], coverage)
        relative = np.column_stack([
            compact[:, COMPACT_NAMES.index(relation)] - compact[:, COMPACT_NAMES.index(own)]
            for relation, own, _ in CONTRASTS
        ])
        enriched = np.concatenate([compact, behavior, relative], axis=1)
        if enriched.shape[1] != 372 or compact.shape[1] != 162 or not np.isfinite(enriched).all():
            raise ValueError("Unexpected feature matrix")
        result = {"ids": db.users.github_user_id.to_numpy(dtype=np.int64)[rows],
                  "compact": compact, "enriched": enriched}
        if labels:
            result.update(db.labels(target, rows))
        return result
