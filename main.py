import pandas as pd
from src.ingestion import IngestData
from src.preprocessing import Preprocess
from src.helper import make_snapshot, make_features
from src.lgb import LGBMClassifier
import lightgbm as lgb
from pathlib import Path
import json

CLARITY_DICTIONARY_PATH = "data/raw/clarity_underwriting_dictionary.csv"


def run_scores(n: int = 10) -> None:
    metadata = json.load(open("deploy/model.json"))
    snapshot = Path(metadata["snapshot_path"])
    booster = lgb.Booster(model_file="deploy/model.txt")
    features = booster.feature_name()

    df = pd.read_parquet(snapshot / "snapshot.parquet")
    sample_df = df.sample(n, random_state=42)
    X = sample_df[features]

    # fix column type
    overallmatch_col = [c for c in features if c.endswith("overallmatchreasoncode")][0]
    X[overallmatch_col] = X[overallmatch_col].astype("category")

    result = pd.DataFrame(
        {
            "risk_score": booster.predict(X),
            "actual": sample_df["target"].values,
        },
        index=sample_df.index,
    )
    result["decision"] = pd.cut(
        result["risk_score"],
        [0, 0.4, 0.7, 1.0],
        labels=["approve", "reivew", "decline"],
    )

    print(result.sort_values("risk_score", ascending=False).to_string())


if "__main__" == __name__:
    loan_df, payment_df, clarity_df = IngestData().load_data()
    loan_preprocessor = Preprocess(loan_df)
    payment_preprocessor = Preprocess(payment_df)
    clarity_preprocessor = Preprocess(clarity_df)

    loan_df = loan_preprocessor.change_date_types(["applicationDate", "originatedDate"])
    payment_df = payment_preprocessor.change_date_types(["paymentDate"])
    clarity_df = clarity_preprocessor.change_clarity_types(
        csv_path=CLARITY_DICTIONARY_PATH
    )

    model_df = loan_preprocessor.extract_funded()
    model_df = loan_preprocessor.add_target()

    model_df = loan_preprocessor.merge_data(
        payment_df=payment_df,
        clarity_df=clarity_df,
        loan_df=loan_df,
    )

    feature_cols = make_features(model_df)

    snapshop_path = make_snapshot(
        model_df[feature_cols + ["target", "applicationDate"]],
        target="target",
    )

    model = LGBMClassifier(
        df=model_df, features=feature_cols, target="target", snapshot_path=snapshop_path
    )

    model, auc, is_saved = model.train_and_eval(optimize=False)

    run_scores()
