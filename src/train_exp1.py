from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from scipy.stats import rankdata
from sklearn.metrics import mean_squared_error
from metric import score
import lightgbm as lgb
from lifelines import NelsonAalenFitter, KaplanMeierFitter, CoxPHFitter
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
import optuna
import warnings

warnings.filterwarnings("ignore")

pd.options.display.max_columns = None

HLA_COLUMNS = [
    "hla_match_a_low", "hla_match_a_high", "hla_match_b_low", "hla_match_b_high",
    "hla_match_c_low", "hla_match_c_high", "hla_match_dqb1_low", "hla_match_dqb1_high",
    "hla_match_drb1_low", "hla_match_drb1_high", "hla_nmdp_6", "hla_low_res_6",
    "hla_high_res_6", "hla_low_res_8", "hla_high_res_8", "hla_low_res_10",
    "hla_high_res_10",
]

class CFG:
    path = Path("data")
    train_path = path / "train.csv"
    test_path = path / "test.csv"
    subm_path = path / "sample_submission.csv"
    color = "#A2574F"
    batch_size = 32768
    early_stop = 200
    penalizer = 0.01
    n_splits = 5
    weights = [1.0, 1.0, 8.0, 4.0, 8.0, 4.0, 6.0, 6.0]
    ctb_params = {
        "loss_function": "RMSE",
        "learning_rate": 0.03,
        "random_state": 42,
        "task_type": "CPU",
        "iterations": 3000,
        "subsample": 0.8,
        "reg_lambda": 10.0,
        "depth": 8,
    }
    lgb_params = {
        "objective": "regression",
        "min_child_samples": 16,
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "extra_trees": True,
        "reg_lambda": 9.0,
        "reg_alpha": 0.2,
        "num_leaves": 72,
        "metric": "rmse",
        "max_depth": 8,
        "device": "cpu",
        "max_bin": 128,
        "verbose": -1,
        "seed": 42,
    }

class FE:
    def __init__(self, batch_size):
        self._batch_size = batch_size

    def load_data(self, path):
        return pl.read_csv(path, batch_size=self._batch_size)

    def recalculate_hla_sums(self, df):
        # Create new aggregated HLA columns
        df = df.with_columns(
            (
                pl.col("hla_match_a_low").fill_null(0) +
                pl.col("hla_match_b_low").fill_null(0) +
                pl.col("hla_match_drb1_high").fill_null(0)
            ).alias("hla_nmdp_6"),
            # Other aggregated features
        )
        return df

    def cast_datatypes(self, df):
        num_cols = [
            "hla_high_res_8", "hla_low_res_8", "hla_high_res_6", "hla_low_res_6",
            "hla_high_res_10", "hla_low_res_10", "hla_match_dqb1_high",
            "hla_match_dqb1_low", "hla_match_drb1_high", "hla_match_drb1_low",
            "hla_nmdp_6", "year_hct", "hla_match_a_high", "hla_match_a_low",
            "hla_match_b_high", "hla_match_b_low", "hla_match_c_high",
            "hla_match_c_low", "donor_age", "age_at_hct", "comorbidity_score",
            "karnofsky_score", "efs", "efs_time",
        ]

        for col in df.columns:
            if col in num_cols:
                df = df.with_columns(pl.col(col).fill_null(-1).cast(pl.Float32))
            else:
                df = df.with_columns(pl.col(col).fill_null("Unknown").cast(pl.String))
        return df.with_columns(pl.col("ID").cast(pl.Int32))

    def apply_fe(self, path):
        df = self.load_data(path)
        df = self.recalculate_hla_sums(df)
        df = self.cast_datatypes(df)
        df = df.to_pandas()
        return df

# Feature Engineering
fe = FE(CFG.batch_size)
train_data = fe.apply_fe(CFG.train_path)
test_data = fe.apply_fe(CFG.test_path)

# Training and Inference
class ModelTraining:
    def __init__(self, train_data, test_data, params, target):
        self.train_data = train_data
        self.test_data = test_data
        self.params = params
        self.target = target

    def train_lgb(self):
        cv = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=42)
        models = []
        oof_preds = np.zeros(len(self.train_data))

        # Convert object columns to categories
        cat_cols = self.train_data.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            self.train_data[col] = self.train_data[col].astype("category")

        for fold, (train_idx, valid_idx) in enumerate(cv.split(self.train_data)):
            X_train, X_valid = self.train_data.iloc[train_idx], self.train_data.iloc[valid_idx]
            y_train, y_valid = self.train_data[self.target].iloc[train_idx], self.train_data[self.target].iloc[valid_idx]

            model = lgb.LGBMRegressor(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="rmse",
                early_stopping_rounds=CFG.early_stop,
                categorical_feature=list(cat_cols),
                verbose=0
            )
            models.append(model)
            oof_preds[valid_idx] = model.predict(X_valid)

        return models, oof_preds

    def train_catboost(self):
        cv = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=42)
        models = []
        oof_preds = np.zeros(len(self.train_data))

        for fold, (train_idx, valid_idx) in enumerate(cv.split(self.train_data)):
            X_train, X_valid = self.train_data.iloc[train_idx], self.train_data.iloc[valid_idx]
            y_train, y_valid = self.train_data[self.target].iloc[train_idx], self.train_data[self.target].iloc[valid_idx]

            model = CatBoostRegressor(**CFG.ctb_params, verbose=0)
            model.fit(
                X_train, y_train,
                eval_set=(X_valid, y_valid),
                early_stopping_rounds=CFG.early_stop
            )
            models.append(model)
            oof_preds[valid_idx] = model.predict(X_valid)

        return models, oof_preds

    def infer(self, models):
        preds = np.mean([model.predict(self.test_data) for model in models], axis=0)
        return preds

# Train and Predict
trainer = ModelTraining(train_data, test_data, CFG.lgb_params, "efs_time")
lgb_models, lgb_oof_preds = trainer.train_lgb()
catboost_models, catboost_oof_preds = trainer.train_catboost()
lgb_test_preds = trainer.infer(lgb_models)
catboost_test_preds = trainer.infer(catboost_models)

# Ensemble
ensemble_preds = 0.6 * lgb_test_preds + 0.4 * catboost_test_preds

# Submission
submission = pd.read_csv(CFG.subm_path)
submission["prediction"] = ensemble_preds
submission.to_csv("submission.csv", index=False)

