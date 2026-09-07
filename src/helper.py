import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple

META_FILENAME = "metadata.json"


def read_metadata(file_path: str | Path) -> dict | None:
    path = Path(file_path)
    if path.is_file():
        with open(path, "r") as f:
            return json.load(f)
    else:
        return None


def save_metadata(metadata: dict, path: str | Path):
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def revoke(model_path: str):
    path = Path(model_path) / "model.json"
    metadata = read_metadata(str(path))

    if metadata:
        snapshot_path = metadata["snapshot_path"]
        path = Path(snapshot_path) / META_FILENAME

        metadata = read_metadata(path)

        if metadata:
            metadata["produced_best_model"] = False
            save_metadata(metadata, path)


def update_snapshot(
    snapshot_path: str, auc: float | None = None, is_best: bool | None = False
):
    path = Path(snapshot_path) / META_FILENAME  # type: ignore
    metadata = read_metadata(path)

    if metadata:
        if auc:
            metadata["auc"] = auc
        metadata["produced_best_model"] = is_best
        save_metadata(metadata, path)


def make_snapshot(
    df: pd.DataFrame,
    target: str,
):
    out_path = Path("data/snapshots") / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path / "snapshot.parquet", index=False)

    json.dump(
        {
            "rows": len(df),
            "columns": len(df.columns),
            "base_rate": float(df[target].mean()),
            "date_range": [
                str(df["applicationDate"].min()),
                str(df["applicationDate"].max()),
            ],
        },
        open(out_path / META_FILENAME, "w"),
        indent=2,
    )

    return str(out_path)


def make_features(df: pd.DataFrame) -> list:
    RISKY = [
        "loanStatus",
        "fpStatus",
        "originated",
        "approved",
        "isFunded",
        "originatedDate",
    ]
    IDS = ["loanId", "anon_ssn", "clarityFraudId", "underwritingid"]
    CONSTANT = [c for c in df.columns if df[c].nunique() <= 1]
    EXCLUDE = set(RISKY + IDS + CONSTANT + ["target", "applicationDate"])

    features = [c for c in df.columns if c not in EXCLUDE]

    return features


def make_split(df: pd.DataFrame, quantile_threshold=0.8) -> Tuple:
    cutoff = df["applicationDate"].quantile(quantile_threshold)
    train = df[df["applicationDate"] <= cutoff]
    test = df[df["applicationDate"] > cutoff]

    return train, test

# def get_mismatch(data: pd.DataFrame, feature_cols: list):
#     metadata = json.load(open("deploy/model.json"))
#     snapshot_path = Path(metadata["snapshot_path"])

#     df = pd.read_parquet(snapshot_path / "snapshot.parquet")

#     before = data[feature_cols].dtypes
#     after = df[feature_cols].dtypes
#     print(before[before != after])