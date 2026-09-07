import lightgbm as lgb
import pandas as pd
from src.helper import make_split, update_snapshot, revoke
from sklearn.metrics import roc_auc_score
from src.tuning import HPT
from typing import Tuple
from pathlib import Path
import json

BEST_MODEL = "deploy"


class LGBMClassifier:
    def __init__(
        self, df: pd.DataFrame, features: list, target: str, snapshot_path: str
    ):
        self.data = df
        self.features = features
        self.target = target
        self.params = self.load_best_params()
        self.snapshot_path = snapshot_path
        Path(BEST_MODEL).mkdir(exist_ok=True)

    @staticmethod
    def load_best_params() -> dict:
        params = {
            "n_estimators": 2000,
            "random_state": 42,
            "n_jobs": -1,
            "learning_rate": 0.020999802041472865,
            "num_leaves": 26,
            "max_depth": 11,
            "min_child_samples": 82,
            "colsample_bytree": 0.6754891174267041,
            "reg_alpha": 6.491708325707309e-07,
            "reg_lambda": 2.819441082308204e-06,
        }

        path = Path("artifacts/best_params.json")

        if path.is_file():
            with path.open():
                content = json.load(open("artifacts/best_params.json"))
                params.update(content["best_params"])

        return params

    def make_model(self):
        model = lgb.LGBMClassifier(**self.params)

        return model

    def train_and_eval(
        self, optimize: bool = False, interactive: bool = False
    ) -> Tuple[lgb.LGBMClassifier, float, bool]:
        if optimize:
            new_params = HPT(self.data, self.features, self.target).tune(
                params=self.params
            )
            self.params.update(new_params)

        train, test = make_split(self.data)
        model = self.make_model()

        model.fit(
            train[self.features],
            train[self.target],
            eval_X=test[self.features],
            eval_y=test[self.target],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
        )

        auc = self.get_auc(model, test)  # type: ignore

        is_saved_as_best = self.evaluate(
            new_model=model, new_auc=auc, test=test, interactive=interactive
        )

        # update snapshot details
        update_snapshot(self.snapshot_path, auc, is_saved_as_best)

        return model, float(auc), is_saved_as_best

    def get_auc(
        self,
        model: lgb.LGBMClassifier | lgb.Booster,
        test: pd.DataFrame,
        is_old: bool = False,
    ) -> float:
        if is_old:
            pred = model.predict(test[self.features])  # type: ignore
        else:
            pred = model.predict_proba(test[self.features])[:, 1]  # type: ignore

        auc = roc_auc_score(test[self.target], pred)  # type: ignore

        return float(auc)

    def evaluate(
        self,
        new_model: lgb.LGBMClassifier,
        new_auc: float,
        test: pd.DataFrame,
        interactive: bool,
    ):
        path = Path(BEST_MODEL) / "model.txt"

        if not path.is_file():
            new_model.booster_.save_model(path)
            print(f"Model saved with AUC: {new_auc}")
            self.save_metadata(snapshot_path=self.snapshot_path, new_auc=new_auc)

            return True
        else:
            old_model = lgb.Booster(model_file=path)
            old_auc = self.get_auc(model=old_model, test=test, is_old=True)  # type: ignore

            if old_auc < 0.70:
                print(f"Warning: AUC {old_auc:.4f} is below 0.70. Investigate drift.")

            if new_auc > old_auc + 0.005:  # new_auc > old_auc + 0.005
                promote = (
                    LGBMClassifier.save_new(new_auc, old_auc) if interactive else True
                )

                if promote:
                    return self.save_model(
                        model=new_model,
                        new_auc=new_auc,
                        old_auc=old_auc,
                        path=str(path),
                        snapshot_path=self.snapshot_path,
                    )
                else:
                    print(f"Kept old model with AUC: {old_auc}")

        return False

    @staticmethod
    def save_model(
        model: lgb.LGBMClassifier | lgb.Booster,
        new_auc: float,
        path: str,
        old_auc: float,
        snapshot_path: str,
    ):
        model.booster_.save_model(path)  # type: ignore
        revoke(BEST_MODEL)
        print(f"Revoked old model with AUC: {old_auc}")
        LGBMClassifier.save_metadata(snapshot_path, new_auc)

        return True

    @staticmethod
    def save_metadata(snapshot_path: str, new_auc: float):
        json.dump(
            {"snapshot_path": snapshot_path},
            open(Path(BEST_MODEL) / "model.json", "w"),
            indent=2,
        )
        print(f"Saved new model with AUC: {new_auc}")

    @staticmethod
    def save_new(new_auc: float, old_auc: float):
        while True:
            result = (
                input(
                    f"New best model: {new_auc} > {old_auc}. Do you want to replace old best? (y/n)"
                )
                .strip()
                .lower()
            )

            if result in ["y", "yes"]:
                return True
            elif result in ["n", "no"]:
                return False
