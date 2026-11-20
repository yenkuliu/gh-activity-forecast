# GitHub Activity Forecasting

Predict six binary activities for each GitHub user over the next seven days: opening a pull request, opening an issue, commenting on an issue or PR, submitting a PR review, commenting on a PR review, and pushing commits. The model combines three TabM classifiers trained on chronological snapshots of eight relational tables.

## Data Source

The event data comes from [GH Archive](https://www.gharchive.org/). The official preprocessing program filters the cohort and creates the anonymized Users, Repositories, PullRequests, Issues, IssueComments, PRReviews, PRReviewComments and Pushes tables. The forecasting code reads these eight Parquet files. [Official data preparation and schema](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/tree/af9f335301d9badb482dfafa10c4534eabac845a)

## Setup

Use Python 3.12 and run these commands from the repository root. The September reference run used PyTorch 2.6.0 with CUDA 12.4. For that GPU setup, install the CUDA wheel before the other requirements. CPU runs can omit the CUDA wheel command and pass `--device cpu` to the model.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124  # NVIDIA GPU
python -m pip install -r requirements.txt

git clone https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git course_tools
git -C course_tools checkout --detach af9f335301d9badb482dfafa10c4534eabac845a
```

## Prepare the data

For the November 22 forecast, the last input hour ends at November 21, 12:00 UTC, which is 20:00 in Taiwan. Wait for that hour to be available before downloading the full interval. For the September benchmark, use the second pair of settings below instead.

```sh
TARGET=2026-11-22
RAW_END=2026-11-21T12:00:00Z

# To reproduce the reference week, use these two values instead:
# TARGET=2026-09-27
# RAW_END=2026-11-08T00:00:00Z

python scripts/download_hours.py --start 2026-02-14 --end "$RAW_END" --out "data/raw-$TARGET"
python course_tools/scripts/preprocess.py --raw "data/raw-$TARGET" --out "data/tables-$TARGET"
```

The download interval is half-open in UTC. Each run’s raw directory should contain only its chosen interval, since the official preprocessor reads every hourly file in that directory. The September release spans 6,408 hours. Existing archives or already-preprocessed tables covering the same interval can be used directly. The downloader reports unavailable files. Upstream omissions within an available file can still affect activity coverage.

## Train

```sh
python -m github_activity train --tables "data/tables-$TARGET/db" \
  --target "$TARGET" --seed 42 --device auto --out "runs/$TARGET"
```

Training creates one directory per member with its weights, normalization statistics, per-epoch metrics and selected epoch. The run also records the split dates, table checksums, ordered feature names, software versions, calibration probabilities, blend weights and six thresholds. Use a new run directory for each fit.

The compact representation has 162 inputs. It combines personal event counts and recency with activity in shared repositories, co-participant activity and incoming interactions. Its time windows are 7, 28, 56 and 112 days, with decay half-lives of 7 and 28 days. The enriched representation has 372 inputs and adds activity gaps, bursts, exposure-adjusted counts and relational contrasts.

| Member | Inputs | Epoch selection |
| --- | ---: | --- |
| Compact TabM | 162 | Lowest masked validation log loss |
| Enriched TabM | 372 | Lowest masked validation log loss |
| Compact TabM | 162 | Highest masked validation F1 score at threshold 0.25 |

Each member has 16 internal predictors and three hidden blocks of width 256. Training minimizes masked binary cross-entropy. Within each member, the sigmoid probabilities are averaged over the 16 predictors. For each activity, a held-out calibration week determines three nonnegative blend weights summing to one and a decision threshold halfway between the learned threshold and 0.25. The architecture uses the [TabM implementation and third-party notices](github_activity/vendor/SOURCES.md).

## Predict the official week

After training for November 22, run the saved model against the official list of test users. The target is November 22 at 00:00 through November 29 at 00:00, Taiwan time. Each row follows the input order and contains the user’s anonymized `Users.github_user_id` and six binary decisions.

```sh
python -m github_activity predict --tables data/tables-2026-11-22/db \
  --model runs/2026-11-22 --users course_tools/submission/test_input.csv \
  --output output_315554009.csv
python course_tools/scripts/evaluate.py --pred output_315554009.csv \
  --input course_tools/submission/test_input.csv
```

If the course updates the user list, pass the same updated CSV to both commands. The check above verifies the submission format and user IDs.

## Reproduce the September benchmark

Prepare the September release and train with `TARGET=2026-09-27` using the commands above. The historical evaluation includes all 9,000 cohort users visible by the feature cutoff. Truth for the seven-day horizon is censored at the exclusive archive observation end.

```sh
python -m github_activity backtest --tables data/tables-2026-09-27/db \
  --model runs/2026-09-27 --observed-through 2026-11-08T00:00:00Z \
  --out runs/reference-evaluation
python course_tools/scripts/evaluate.py --pred runs/reference-evaluation/predictions.csv \
  --truth runs/reference-evaluation/truth.csv
```

| Reference week, Taiwan time | Seed | Macro-F1 | Micro-F1 | Score |
| --- | ---: | ---: | ---: | ---: |
| Sep 27–Oct 4, 2026 | 42 | 51.57% | 61.24% | **56.41%** |

Score is the mean of macro-F1 and micro-F1. These are historical results for this release and training environment. The aggregate metrics and fitted decision parameters are in the [reference record](reference/2026-09-27.json). Training regenerates the model weights. Backend or library differences can change the fit.

### Chronological protocol

Features use the 112 days ending four hours before each Sunday target. Both event time and first visibility must precede the cutoff. Training, epoch selection and threshold calibration use earlier label horizons. The loss masks the Push target when any UTC day touched by its seven-day window lacks an observed Push by the relevant fitting cutoff.

For the September reference, ten training weeks start between June 7 and August 9. Epoch selection uses the weeks starting August 23 and August 30. The September 13 week fits the blend and thresholds. September 27 is reserved for evaluation. The same procedure rolls forward for the November forecast.
