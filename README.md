# Gridlock Hackathon 2.0 — Traffic Demand Prediction

A gradient-boosting pipeline that predicts traffic demand for geohash locations at 15-minute intervals, built for the Gridlock Hackathon 2.0.

**Final leaderboard score:** `172` <!-- replace with your score -->

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Key Insight](#key-insight)
4. [Approach](#approach)
5. [Feature Engineering](#feature-engineering)
6. [Model and Training](#model-and-training)
7. [Post-processing](#post-processing)
8. [Validation Strategy](#validation-strategy)
9. [Repository Structure](#repository-structure)
10. [How to Run](#how-to-run)
11. [Tech Stack](#tech-stack)

---

## Problem Statement
Cities increasingly use AI to understand and reduce traffic congestion. The task is to predict traffic **demand** (a value between 0 and 1) for specific locations at specific times, which can feed analyses of travel patterns and demand forecasting.

**Evaluation metric:** `score = max(0, 100 * R²(actual, predicted))`

## Dataset

| File | Shape | Description |
|---|---|---|
| `train.csv` | 77,299 × 11 | Labelled data for Day 48 (full day) and Day 49 (hours 0–2 only) |
| `test.csv` | 41,778 × 10 | Day 49 from 2:15 to 13:45, no labels |
| `sample_submission.csv` | 5 × 2 | Submission format (`Index`, `demand`) |

**Columns**

| Column | Description |
|---|---|
| `Index` | Unique ID of the data point |
| `geohash` | Geographic location code |
| `day` | Day of the record |
| `timestamp` | Time of the record (15-minute slots) |
| `RoadType` | Type of road nearby |
| `NumberofLanes` | Number of roads/lanes at the location |
| `LargeVehicles` | Whether large vehicles are permitted |
| `Landmarks` | Whether landmarks are nearby |
| `Temperature` | Temperature at the location |
| `Weather` | Weather at the location |
| `demand` | Target: traffic demand at the timestamp |

**Structure of the split.** The test set is not a random sample. It is the continuation of Day 49 after the last training slot, so the problem is closer to short-horizon forecasting than to ordinary tabular regression.

## Key Insight
Two facts about the data drive the whole solution:

1. **Day 48 is a near-complete structural proxy for Day 49.** About 89% of test rows have an exact Day 48 match on (geohash, timestamp), and about 96% match on (geohash, hour).
2. **Day 49 does not simply repeat Day 48.** The Day 49 / Day 48 demand ratio changes over the morning. It is roughly 2.6 at hour 0, 1.8 at hour 1 and 1.5 at hour 2, so a single constant scaling factor is wrong for later hours.

So instead of predicting demand from scratch, the model starts from Day 48 and learns how Day 49 is drifting away from it.

## Approach
1. Look up the Day 48 demand for the same (geohash, timestamp).
2. For each geohash, compute the Day 49 / Day 48 ratio at hours 0, 1 and 2.
3. Fit a linear trend to those three ratios and extrapolate it to each target hour (`ratio_at_hour`).
4. Create scaled estimates, such as Day 48 demand × `ratio_at_hour`, and feed them to LightGBM along with temporal, spatial and road features.
5. Train a 5-seed × 5-fold LightGBM ensemble with Day 49 rows upweighted.
6. Apply a light blend with the rule-based estimate.

## Feature Engineering
57 features in total, grouped as follows.

**Temporal**
- `hour`, `minute`, `time_slot` (0–95), `time_of_day`
- Cyclical sin/cos encodings of hour and time slot
- `is_peak` (07:00–10:00) and `is_night` (22:00–05:00) flags, `day`

**Spatial**
- Latitude and longitude decoded from the geohash
- Label-encoded geohash and its 3-, 4- and 5-character prefixes

**Road and environment**
- Large-vehicle permission, landmark presence, road type (ordinal), weather (ordinal), number of lanes
- Temperature, with missing values filled by the per-geohash median

**Cross-day core features (strongest signals)**
- `d48_demand`: exact Day 48 lookup at (geohash, timestamp), with an hourly fallback
- `d48_hr_curr`, `d48_hr_prev`: Day 48 hourly demand
- `d49_same_hr`, `d49_prev_hr`: Day 49 hourly demand where observed

**Ratio features**
- `ratio_at_hour`: per-geohash ratio trend extrapolated to the target hour (the key signal)
- `ratio_at_h2`: the ratio at hour 2, the most recent observation
- `ratio_slope`: direction and speed of the ratio trend
- `day_ratio`, `geo4_ratio`: early-morning ratio at geohash level and geo4 level, used as fallbacks for sparse geohashes

**Scaled estimates**
- `d48_day_ratio_pred`, `d48_hour_ratio_pred`, `d48_geo4_ratio_pred`
- `geo_hour_ratio_pred`, `geo4_hour_ratio_pred`, `geo_hour_hratio_pred`

**Slot-level smoothing on Day 48**
- `d48_prev_slot`, `d48_next_slot`: neighbouring 15-minute slots
- `d48_roll3_mean`: 3-slot rolling mean, a more stable baseline than a single slot
- `d48_roll3_hratio_pred`: rolling mean × `ratio_at_hour`

**Day 49 anchors**
- `d49_at_2_0`: demand at exactly 2:00, the slot closest to the test start
- `d49_last2_mean`: mean of the 1:45 and 2:00 slots
- `d49_trend`, `d49_last_demand`, `d49_extrapolated`: trend over the observed Day 49 hours projected forward

**Historical statistics from Day 48**
- `geo_hour_mean/std/med/q25/q75` per (geohash, hour)
- `geo_d48_mean/std/med` per geohash
- `geo4_hour_mean`, `geo3_hour_mean`

**Structural features**
- `d48_day_slope`: linear slope of Day 48 demand across the day, separating rising from falling locations
- `geo4_neighbor_mean`: mean Day 48 demand of other geohashes in the same geo4 cell, which helps sparse locations

## Model and Training
- **Algorithm:** LightGBM (`LGBMRegressor`), RMSE objective
- **Key parameters:** `learning_rate=0.015`, `num_leaves=511`, `n_estimators=4000`, `min_child_samples=20`, `feature_fraction=0.8`, `bagging_fraction=0.8` (every 5 iterations), `reg_alpha=0.05`, `reg_lambda=0.1`, early stopping after 150 rounds
- **Sample weighting:** Day 49 rows are weighted 9×. They are only about 10% of the training data but match the conditions of the test set.
- **Ensemble:** 5-fold shuffled CV repeated with seeds `[42, 7, 123, 314, 999]`, giving 25 models. Test predictions are the mean of all 25.

## Post-processing
```
final_pred = 0.95 * model_pred + 0.05 * d48_hour_ratio_pred
```
Both terms are clipped to [0, 1]. The blend is used only if it improves the Day 49 out-of-fold R²; otherwise the raw ensemble output is used. The small weight anchors extreme extrapolations without distorting the model's output.

## Validation Strategy
The overall out-of-fold R² is inflated, because roughly 90% of the training rows are Day 48 rows that sit close to their own lookup features. The notebook therefore reports both:

- `R²_all`: out-of-fold R² on all training rows
- `R²_d49`: out-of-fold R² on Day 49 rows only, which is the closest proxy for the leaderboard

Model decisions were judged on `R²_d49`.

## Repository Structure
```
.
├── GridlockHackathon.ipynb   # full pipeline: features, training, prediction
├── submission.csv            # final predictions (Index, demand)
├── APPROACH.txt              # detailed written approach
├── requirements.txt          # Python dependencies
├── .gitignore
└── data/                     # place train.csv and test.csv here (git-ignored)
```

## How to Run
1. Clone the repository and install the dependencies:
   ```bash
   git clone https://github.com/<your-username>/gridlock-traffic-demand.git
   cd gridlock-traffic-demand
   pip install -r requirements.txt
   ```
2. Download `train.csv` and `test.csv` from the hackathon page. The notebook reads them from its working directory, so put them in the same folder as the notebook (or edit the paths in the data-loading cell).
3. Open and run all cells:
   ```bash
   jupyter notebook GridlockHackathon.ipynb
   ```
4. The notebook writes `submission.csv` with columns `Index` and `demand`, in the format required by the hackathon.

**Runtime note:** the full ensemble trains 25 models with a low learning rate and a high number of trees, so a complete run takes a while. For a quick test, reduce the number of seeds in the `SEEDS` list.

## Tech Stack
Python · pandas · NumPy · LightGBM · scikit-learn · pygeohash · matplotlib
