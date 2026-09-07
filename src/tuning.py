import lightgbm as lgb
import pandas as pd
import optuna
from sklearn.metrics import roc_auc_score
from src.helper import make_split
from pathlib import Path
import json


class HPT:
    def __init__(self, df: pd.DataFrame, features: list, target: str) -> None:
        self.df = df
        self.features = features
        self.target = target
        self.train, self.test = make_split(self.df)
        Path("artifacts").mkdir(exist_ok=True)

    def objective(self, trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.001, 0.05, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(
            self.train[self.features],
            self.train[self.target],
            eval_X=self.test[self.features],
            eval_y=self.test[self.target],
            callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)],
        )

        preds = model.predict_proba(self.test[self.features])[:, 1]  # type: ignore
        auc = roc_auc_score(self.test[self.target], preds)

        return auc

    def tune(self, trials=20, params: dict | None = None) -> dict:
        study = optuna.create_study(direction="maximize")

        if params:
            study.enqueue_trial(params)

        study.optimize(self.objective, n_trials=trials, timeout=1200)  # type: ignore

        print("Number of finished trials:", len(study.trials))
        print("Best trial:", study.best_trial.params)

        json.dump(
            {"best_params": study.best_params, "best_auc": study.best_value},
            open("artifacts/best_params.json", "w"),
            indent=2,
        )

        return study.best_params
