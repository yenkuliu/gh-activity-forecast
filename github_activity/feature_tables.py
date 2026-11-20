"""Temporal joins over the eight anonymized GitHub activity tables."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


DAY = 86_400
WEEK = 7 * DAY
GAP = 4 * 3_600
MISSING_TIME = np.iinfo(np.int64).max
EVENTS = ("open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push")

PRIMARY_KEYS = {
    "Users": "user_id", "Repositories": "repo_id", "PullRequests": "pr_id",
    "Issues": "issue_id", "IssueComments": "comment_id", "PRReviews": "review_id",
    "PRReviewComments": "comment_id", "Pushes": "push_id",
}

# (table, actor, event time, first observation). Order is the submission column order.
EVENT_SOURCES = (
    ("PullRequests", "author_id", "created_at", "first_seen_at"),
    ("Issues", "author_id", "created_at", "first_seen_at"),
    ("IssueComments", "author_id", "created_at", "created_at"),
    ("PRReviews", "reviewer_id", "submitted_at", "first_seen_at"),
    ("PRReviewComments", "author_id", "created_at", "created_at"),
    ("Pushes", "user_id", "created_at", "created_at"),
)
PARENT_LINKS = (
    (2, "IssueComments", "issue_id", "Issues"),
    (3, "PRReviews", "pr_id", "PullRequests"),
    (4, "PRReviewComments", "pr_id", "PullRequests"),
)


def target_seconds(date):
    """Convert a Taiwan-local forecast date to a Unix timestamp in seconds."""
    date = pd.Timestamp(date)
    if date.tzinfo is None:
        date = date.tz_localize("Asia/Taipei")
    return int(date.timestamp())


def _timestamps(series):
    seconds = series.to_numpy(dtype="datetime64[s]").astype(np.int64)
    seconds[series.isna().to_numpy()] = MISSING_TIME
    return seconds


class ActivityTables:
    def __init__(self, directory):
        directory = Path(directory)
        columns = {
            "Users": {"github_user_id", "is_cohort", "first_seen_at"},
            "Repositories": {"first_seen_at"},
        }
        table_keys = {}
        for table, actor, time, seen in EVENT_SOURCES:
            columns[table] = {actor, "repo_id", time, seen}
            table_keys[table] = {actor: "Users", "repo_id": "Repositories"}
        columns["Issues"].add("is_pull_request")
        for _, table, key, parent in PARENT_LINKS:
            columns[table].add(key)
            table_keys[table][key] = parent
        self.tables = {}
        for table, required in columns.items():
            pk = PRIMARY_KEYS[table]
            df = pd.read_parquet(directory / f"{table}.parquet", columns=sorted(required | {pk}))
            if not np.array_equal(df[pk].to_numpy(), np.arange(len(df))):
                raise ValueError(f"{table}.{pk} must be the zero-based row index")
            for field in required & {"created_at", "submitted_at", "first_seen_at"}:
                df[field] = _timestamps(df[field])
            for field in table_keys.get(table, {}):
                df[field] = df[field].fillna(-1).astype(np.int32)
            self.tables[table] = df
        for table, keys in table_keys.items():
            for field, parent in keys.items():
                values = self.tables[table][field].to_numpy()
                if np.any((values < -1) | (values >= len(self.tables[parent]))):
                    raise ValueError(f"{table}.{field} references an invalid {parent} row")

        self.users = self.tables["Users"]
        self.user_seen = self.users.first_seen_at.to_numpy()
        self.repo_seen = self.tables["Repositories"].first_seen_at.to_numpy()
        self.cohort = np.flatnonzero(self.users.is_cohort.fillna(False).to_numpy(bool))
        self.n_users = len(self.users)
        self.n_repos = len(self.repo_seen)
        self.events = self._build_events()

    def _build_events(self):
        groups = []
        for kind, (name, actor, time, seen) in enumerate(EVENT_SOURCES):
            source = self.tables[name]
            events = pd.DataFrame({
                "u": source[actor], "r": source.repo_id, "time": source[time],
                "seen": source[seen], "kind": np.int8(kind),
                "row": np.arange(len(source), dtype=np.int32),
            })
            if name == "Issues":
                events = events[~source.is_pull_request.to_numpy()]
            groups.append(events[(events.u >= 0) & (events.r >= 0) & (events.time < MISSING_TIME)])
        return pd.concat(groups, ignore_index=True).sort_values("time", kind="stable", ignore_index=True)

    def history(self, cutoff):
        events = self.events
        available = (events.time < cutoff) & (events.time >= cutoff - 112 * DAY)
        available &= (events.seen < cutoff) & (self.user_seen[events.u] < cutoff)
        available &= self.repo_seen[events.r] < cutoff
        return events[available]

    def counts(self, events):
        keys = events.u.to_numpy() * 6 + events.kind.to_numpy()
        return np.bincount(keys, minlength=self.n_users * 6).reshape(self.n_users, 6).astype(np.float32)

    def _incoming(self, events, cutoff):
        counts = np.zeros((self.n_users, 6), dtype=np.float32)
        for kind, table, field, parent_name in PARENT_LINKS:
            row = events.loc[events.kind == kind, "row"].to_numpy()
            keys = self.tables[table][field].to_numpy()[row]
            parent = self.tables[parent_name].iloc[keys[keys >= 0]]
            authors = parent.author_id.to_numpy()
            authors = authors[(authors >= 0) & (parent.first_seen_at.to_numpy() < cutoff)]
            np.add.at(counts[:, kind], authors[self.user_seen[authors] < cutoff], 1)
        return np.log1p(counts)

    def relations(self, events, user_counts, cutoff):
        """Repository events, co-user events, and responses to a user's issue or PR."""
        if events.empty:
            return [np.zeros((self.n_users, n), dtype=np.float32) for n in (7, 7, 6)]
        incidence = sparse.coo_matrix(
            (np.ones(len(events), np.float32), (events.u, events.r)),
            shape=(self.n_users, self.n_repos),
        ).tocsr()
        incidence.data[:] = 1
        degree = np.asarray(incidence.sum(axis=1)).ravel()
        people = np.asarray(incidence.sum(axis=0)).ravel()
        repo_keys = events.r.to_numpy() * 6 + events.kind.to_numpy()
        repo_counts = np.bincount(repo_keys, minlength=self.n_repos * 6).reshape(self.n_repos, 6)
        repo = np.maximum(incidence @ repo_counts - user_counts, 0) / np.maximum(degree[:, None], 1)
        paths = np.maximum(np.asarray(incidence @ people).ravel() - degree, 0)
        other = incidence @ (incidence.T @ user_counts) - degree[:, None] * user_counts
        peer = np.maximum(other, 0) / np.maximum(paths[:, None], 1)
        return [
            np.column_stack([np.log1p(repo), np.log1p(degree)]).astype(np.float32),
            np.column_stack([np.log1p(peer), np.log1p(paths)]).astype(np.float32),
            self._incoming(events, cutoff),
        ]

    def labels(self, target, rows):
        """Earliest observed positive per user/activity, plus Push-day availability."""
        positive_seen = np.full((self.n_users, 6), MISSING_TIME, dtype=np.int64)
        future = self.events[(self.events.time >= target) & (self.events.time < target + WEEK)]
        seen = np.maximum.reduce([
            future.seen.to_numpy(), self.user_seen[future.u], self.repo_seen[future.r],
        ])
        np.minimum.at(positive_seen, (future.u, future.kind), seen)
        # All UTC days touched by the target week must have an observed Push.
        # Include partial UTC days at the boundaries of a Taiwan-local week.
        push = self.events[self.events.kind == 5]
        day_first_seen = push.groupby(push.time // DAY).seen.min()
        days = range(target // DAY, (target + WEEK - 1) // DAY + 1)
        push_known_by = max(day_first_seen.get(day, MISSING_TIME) for day in days)
        return {
            "positive_seen": positive_seen[rows],
            "push_known_by": np.full(len(rows), push_known_by, dtype=np.int64),
        }
