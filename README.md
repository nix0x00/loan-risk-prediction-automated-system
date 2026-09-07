# Loan Risk Model & Automated Pipeline

Predicts the risk of a loan application defaulting, and wraps that model in an
automated pipeline that retrains, evaluates against the selected model, and promotes
a new model only when it demonstrably beats the one in production.

## Directory structure

```
.
├── main.ipynb                  analysis notebook (EDA → target → features → model)
├── main.py                     pipeline entrypoint, runs end-to-end
├── src/
│   ├── ingestion.py            IngestData — locates and loads the three parquet sources,
│   │                           fails the run if any file is missing
│   ├── preprocessing.py        EDA        — reusable profiler used in the notebook
│   │                           Preprocess — date/dtype casting, target definition,
│   │                                        funded-loan filtering, the merging
│   ├── helper.py               snapshot creation and metadata, feature list construction,
│   │                           time-based train/test split
│   ├── lgb.py                  LGBMClassifier — training, evaluation
│   │                           comparison and promotion
│   └── tuning.py               HPT — Optuna hyperparameter search, persists best params
├── data/
│   ├── raw/                    source parquet files + clarity csv 
│   └── snapshots/<timestamp>/  immutable training datasets
│       ├── snapshot.parquet    the modelling dataset for that run
│       └── metadata.json       rows, base rate, date range, AUC, promotion outcome
├── deploy/
│   ├── model.txt               promoted model, native LightGBM format
│   └── model.json              pointer to the snapshot that produced it
├── artifacts/
│   └── best_params.json        best hyperparameters found by Optuna
└── pyproject.toml / uv.lock    pinned environment
```

## `main.html / main.ipynb`

The analysis and the reasoning behind the model. Runs in order:

1. **EDA** across all three files — data quality register, structural vs defective nulls.
2. **Funnel analysis** — `originated` / `approved` / `isFunded`, and the `loanStatus`
   taxonomy that classifies each status as good / bad / unresolved.
3. **Target definition** — funded loans with a resolved outcome (29,220 rows, ~61% bad rate).
4. **Feature building** — loan attributes, clarity underwriting variables, and applicant
   payment history restricted by an as-of rule to avoid leakage.
5. **Model** — LightGBM with a time-based split, evaluated on AUC, Gini, KS.
6. **Limitations** — selection bias, right-censoring, and unused signal.

## `docs/XYZ-Corp.pdf`
The architecture diagram and design rationale.



## `main.py`

The same steps as the notebook, but as a repeatable pipeline. One run does:

```
ingest → validate → preprocess → join → build features → write snapshot
      → train → evaluate vs chosen best → promote (or keep) → score sample
```

Run it:

```bash
python main.py
```

Behaviour:

- Writes a new timestamped snapshot under `data/snapshots/` on every run, with its
  `metadata.json`. Snapshots are never overwritten.
- Trains on the snapshot, using the hyperparameters in `artifacts/best_params.json`
  when present and documented defaults otherwise.
- Compares the new model against the model in `deploy/` on the same held-out period.
  Promotion requires an AUC improvement of at least 0.005; otherwise the incumbent stays
  and the decision is logged.
- Warns if the incumbent's AUC on fresh data falls below the acceptable floor.
- Finishes by scoring a sample of applications with the deployed model, showing the risk
  score and the approve / review / decline band.

Hyperparameter search is off by default — it is the most expensive stage and is intended
to run on a slower flow than retraining.
