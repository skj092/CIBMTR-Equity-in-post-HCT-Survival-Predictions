from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from scipy.stats import rankdata
from metric import score
import lightgbm as lgb
from lifelines import NelsonAalenFitter
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

pd.options.display.max_columns = None

HLA_COLUMNS = [
    "hla_match_a_low",
    "hla_match_a_high",
    "hla_match_b_low",
    "hla_match_b_high",
    "hla_match_c_low",
    "hla_match_c_high",
    "hla_match_dqb1_low",
    "hla_match_dqb1_high",
    "hla_match_drb1_low",
    "hla_match_drb1_high",
    "hla_nmdp_6",
    "hla_low_res_6",
    "hla_high_res_6",
    "hla_low_res_8",
    "hla_high_res_8",
    "hla_low_res_10",
    "hla_high_res_10",
]


class CFG:
    path = Path("data")
    train_path = path / "train.csv"
    test_path = path / "test.csv"
    subm_path = path / "sample_submission.csv"
    color = "#A2574F"
    batch_size = 32768
    early_stop = 500
    penalizer = 0.005
    n_splits = 7
    weights = [1.5, 1.0, 8.0, 4.0, 8.0, 4.0, 7.0, 6.5]
    ctb_params = {
        "loss_function": "RMSE",
        "learning_rate": 0.02,
        "random_state": 42,
        "task_type": "CPU",
        "num_trees": 6000,
        "subsample": 0.85,
        "reg_lambda": 8.0,
        "depth": 9,
        "min_data_in_leaf": 20,
        "max_bin": 256
    }
    lgb_params = {
        "objective": "regression",
        "min_child_samples": 32,
        "num_iterations": 6000,
        "learning_rate": 0.03,
        "extra_trees": True,
        "reg_lambda": 8.0,
        "reg_alpha": 0.1,
        "num_leaves": 64,
        "metric": "rmse",
        "max_depth": 8,
        "device": "cpu",
        "max_bin": 128,
        "verbose": -1,
        "seed": 42,
    }
    # Parameters for the first CatBoost model with Cox loss function
    cox1_params = {
        "grow_policy": "Depthwise",
        "min_child_samples": 8,
        "loss_function": "Cox",
        "learning_rate": 0.03,
        "random_state": 42,
        "task_type": "CPU",
        "num_trees": 6000,
        "subsample": 0.85,
        "reg_lambda": 8.0,
        "depth": 8,
    }
    # Parameters for the second CatBoost model with Cox loss function
    cox2_params = {
        "grow_policy": "Lossguide",
        "loss_function": "Cox",
        "learning_rate": 0.03,
        "random_state": 42,
        "task_type": "CPU",
        "num_trees": 6000,
        "subsample": 0.85,
        "reg_lambda": 8.0,
        "num_leaves": 32,
        "depth": 8,
    }


class FE:

    def __init__(self, batch_size):
        self._batch_size = batch_size

    def load_data(self, path):
        return pl.read_csv(path, batch_size=self._batch_size)

    def recalculate_hla_sums(self, df):
        df = df.with_columns(
            (
                pl.col("hla_match_a_low").fill_null(0)
                + pl.col("hla_match_b_low").fill_null(0)
                + pl.col("hla_match_drb1_high").fill_null(0)
            ).alias("hla_nmdp_6"),
            (
                pl.col("hla_match_a_low").fill_null(0)
                + pl.col("hla_match_b_low").fill_null(0)
                + pl.col("hla_match_drb1_low").fill_null(0)
            ).alias("hla_low_res_6"),
            (
                pl.col("hla_match_a_high").fill_null(0)
                + pl.col("hla_match_b_high").fill_null(0)
                + pl.col("hla_match_drb1_high").fill_null(0)
            ).alias("hla_high_res_6"),
            (
                pl.col("hla_match_a_low").fill_null(0)
                + pl.col("hla_match_b_low").fill_null(0)
                + pl.col("hla_match_c_low").fill_null(0)
                + pl.col("hla_match_drb1_low").fill_null(0)
            ).alias("hla_low_res_8"),
            (
                pl.col("hla_match_a_high").fill_null(0)
                + pl.col("hla_match_b_high").fill_null(0)
                + pl.col("hla_match_c_high").fill_null(0)
                + pl.col("hla_match_drb1_high").fill_null(0)
            ).alias("hla_high_res_8"),
            (
                pl.col("hla_match_a_low").fill_null(0)
                + pl.col("hla_match_b_low").fill_null(0)
                + pl.col("hla_match_c_low").fill_null(0)
                + pl.col("hla_match_drb1_low").fill_null(0)
                + pl.col("hla_match_dqb1_low").fill_null(0)
            ).alias("hla_low_res_10"),
            (
                pl.col("hla_match_a_high").fill_null(0)
                + pl.col("hla_match_b_high").fill_null(0)
                + pl.col("hla_match_c_high").fill_null(0)
                + pl.col("hla_match_drb1_high").fill_null(0)
                + pl.col("hla_match_dqb1_high").fill_null(0)
            ).alias("hla_high_res_10"),
        )

        return df

    def cast_datatypes(self, df):
        num_cols = [
            "hla_high_res_8",
            "hla_low_res_8",
            "hla_high_res_6",
            "hla_low_res_6",
            "hla_high_res_10",
            "hla_low_res_10",
            "hla_match_dqb1_high",
            "hla_match_dqb1_low",
            "hla_match_drb1_high",
            "hla_match_drb1_low",
            "hla_nmdp_6",
            "year_hct",
            "hla_match_a_high",
            "hla_match_a_low",
            "hla_match_b_high",
            "hla_match_b_low",
            "hla_match_c_high",
            "hla_match_c_low",
            "donor_age",
            "age_at_hct",
            "comorbidity_score",
            "karnofsky_score",
            "efs",
            "efs_time",
        ]

        for col in df.columns:
            if col in num_cols:
                df = df.with_columns(pl.col(col).fill_null(-1).cast(pl.Float32))
            else:
                df = df.with_columns(pl.col(col).fill_null("Unknown").cast(pl.String))
        return df.with_columns(pl.col("ID").cast(pl.Int32))

    def info(self, df):
        print(f"\nShape of dataframe: {df.shape}")
        mem = df.memory_usage().sum() / 1024**2
        print("Memory usage: {:.2f} MB\n".format(mem))
        # print(df.head())

    def apply_fe(self, path):
        df = self.load_data(path)
        df = self.recalculate_hla_sums(df)
        df = self.cast_datatypes(df)
        df = df.to_pandas()
        self.info(df)
        cat_cols = [col for col in df.columns if df[col].dtype == pl.String]
        return df, cat_cols


fe = FE(CFG.batch_size)
train_data, cat_cols = fe.apply_fe(CFG.train_path)
test_data, _ = fe.apply_fe(CFG.test_path)


class Targets:
    def __init__(self, data, cat_cols, penalizer, n_splits):
        self.data = data
        self.cat_cols = cat_cols
        self._length = len(self.data)
        self._penalizer = penalizer
        self._n_splits = n_splits

    def _prepare_cv(self):
        oof_preds = np.zeros(self._length)
        cv = KFold(n_splits=self._n_splits, shuffle=True, random_state=42)
        return cv, oof_preds

    def validate_model(self, preds, title):
        y_true = self.data[["ID", "efs", "efs_time", "race_group"]].copy()
        y_pred = self.data[["ID"]].copy()
        y_pred["prediction"] = preds
        c_index_score = score(y_true.copy(), y_pred.copy(), "ID")
        print(f"Overall Stratified C-Index Score for {title}: {c_index_score:.4f}")

    def create_target1(self):
        cv, oof_preds = self._prepare_cv()
        # Apply one hot encoding to categorical columns
        data = pd.get_dummies(self.data, columns=self.cat_cols, drop_first=True).drop(
            "ID", axis=1
        )
        for train_index, valid_index in cv.split(data):
            train_data = data.iloc[train_index]
            valid_data = data.iloc[valid_index]

            # Drop constant columns if they exist
            train_data = train_data.loc[:, train_data.nunique() > 1]
            valid_data = valid_data[train_data.columns]

            cph = CoxPHFitter(penalizer=self._penalizer)
            cph.fit(train_data, duration_col="efs_time", event_col="efs")

            oof_preds[valid_index] = cph.predict_partial_hazard(valid_data)

        self.data["target1"] = oof_preds
        self.validate_model(oof_preds, "Cox")

        return self.data

    def create_target2(self):
        cv, oof_preds = self._prepare_cv()
        for train_index, valid_index in cv.split(self.data):

            train_data = self.data.iloc[train_index]
            valid_data = self.data.iloc[valid_index]

            kmf = KaplanMeierFitter()
            kmf.fit(durations=train_data["efs_time"], event_observed=train_data["efs"])

            oof_preds[valid_index] = kmf.survival_function_at_times(
                valid_data["efs_time"]
            ).values

        self.data["target2"] = oof_preds
        self.validate_model(oof_preds, "Kaplan-Meier")

        return self.data

    def create_target3(self):
        cv, oof_preds = self._prepare_cv()
        for train_index, valid_index in cv.split(self.data):

            train_data = self.data.iloc[train_index]
            valid_data = self.data.iloc[valid_index]

            naf = NelsonAalenFitter()
            naf.fit(durations=train_data["efs_time"], event_observed=train_data["efs"])

            oof_preds[valid_index] = -naf.cumulative_hazard_at_times(
                valid_data["efs_time"]
            ).values

        self.data["target3"] = oof_preds
        self.validate_model(oof_preds, "Nelson-Aalen")

        return self.data

    def create_target4(self):

        self.data["target4"] = self.data.efs_time.copy()
        self.data.loc[self.data.efs == 0, "target4"] *= -1

        return self.data


class MD:
    def __init__(self, color, data, cat_cols, early_stop, penalizer, n_splits):
        self.targets = Targets(data, cat_cols, penalizer, n_splits)
        self.data = data
        self.cat_cols = cat_cols
        self._early_stop = early_stop

    def create_targets(self):

        self.data = self.targets.create_target1()
        self.data = self.targets.create_target2()
        self.data = self.targets.create_target3()
        self.data = self.targets.create_target4()

        return self.data

    def train_model(self, params, target, title):
        feature_importance = pd.DataFrame()
        for col in self.cat_cols:
            self.data[col] = self.data[col].astype("category")
        X = self.data.drop(
            ["ID", "efs", "efs_time", "target1", "target2", "target3", "target4"],
            axis=1,
        )
        y = self.data[target]
        models, fold_scores = [], []
        cv, oof_preds = self.targets._prepare_cv()
        for fold, (train_index, valid_index) in enumerate(cv.split(X, y)):

            X_train = X.iloc[train_index]
            X_valid = X.iloc[valid_index]

            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]

            if title.startswith("LightGBM"):
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric="rmse",
                    callbacks=[
                        lgb.early_stopping(self._early_stop, verbose=0),
                        lgb.log_evaluation(0),
                        lgb.record_evaluation({}),

                    ],
                )

            elif title.startswith("CatBoost"):
                model = CatBoostRegressor(
                    **params, verbose=0, cat_features=self.cat_cols
                )
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_valid, y_valid),
                    early_stopping_rounds=self._early_stop,
                    verbose=0,
                )

            models.append(model)
            oof_preds[valid_index] = model.predict(X_valid)
            y_true_fold = self.data.iloc[valid_index][
                ["ID", "efs", "efs_time", "race_group"]
            ].copy()
            y_pred_fold = self.data.iloc[valid_index][["ID"]].copy()
            y_pred_fold["prediction"] = oof_preds[valid_index]
            fold_score = score(y_true_fold, y_pred_fold, "ID")
            fold_scores.append(fold_score)

        self.targets.validate_model(oof_preds, title)
        return models, oof_preds

    def infer_model(self, data, models):
        data = data.drop(["ID"], axis=1)
        for col in self.cat_cols:
            data[col] = data[col].astype("category")
        return np.mean([model.predict(data) for model in models], axis=0)


md = MD(CFG.color, train_data, cat_cols, CFG.early_stop, CFG.penalizer, CFG.n_splits)

train_data = md.create_targets()
fe.info(train_data)

lgb1_models, lgb1_oof_preds = md.train_model(
    CFG.lgb_params, target="target1", title="LightGBM"
)
import sys; sys.exit()

ctb1_models, ctb1_oof_preds = md.train_model(
    CFG.ctb_params, target="target1", title="CatBoost"
)

ctb1_preds = md.infer_model(test_data, ctb1_models)
lgb1_preds = md.infer_model(test_data, lgb1_models)

ctb2_models, ctb2_oof_preds = md.train_model(
    CFG.ctb_params, target="target2", title="CatBoost"
)
lgb2_models, lgb2_oof_preds = md.train_model(
    CFG.lgb_params, target="target2", title="LightGBM"
)

ctb2_preds = md.infer_model(test_data, ctb2_models)
lgb2_preds = md.infer_model(test_data, lgb2_models)


ctb3_models, ctb3_oof_preds = md.train_model(
    CFG.ctb_params, target="target3", title="CatBoost"
)
lgb3_models, lgb3_oof_preds = md.train_model(
    CFG.lgb_params, target="target3", title="LightGBM"
)

ctb3_preds = md.infer_model(test_data, ctb3_models)
lgb3_preds = md.infer_model(test_data, lgb3_models)


cox1_models, cox1_oof_preds = md.train_model(
    CFG.cox1_params, target="target4", title="CatBoost"
)
cox2_models, cox2_oof_preds = md.train_model(
    CFG.cox2_params, target="target4", title="CatBoost"
)

cox1_preds = md.infer_model(test_data, cox1_models)
cox2_preds = md.infer_model(test_data, cox2_models)

oof_preds = [
    ctb1_oof_preds,
    lgb1_oof_preds,
    ctb2_oof_preds,
    lgb2_oof_preds,
    ctb3_oof_preds,
    lgb3_oof_preds,
    cox1_oof_preds,
    cox2_oof_preds,
]

ranked_oof_preds = np.array([rankdata(p) for p in oof_preds])
ensemble_oof_preds = np.dot(CFG.weights, ranked_oof_preds)

md.targets.validate_model(ensemble_oof_preds, "Ensemble Model")

preds = [
    ctb1_preds,
    lgb1_preds,
    ctb2_preds,
    lgb2_preds,
    ctb3_preds,
    lgb3_preds,
    cox1_preds,
    cox2_preds,
]

ranked_preds = np.array([rankdata(p) for p in preds])

ensemble_preds = np.dot(CFG.weights, ranked_preds)

subm_data = pd.read_csv(CFG.subm_path)
subm_data["prediction"] = ensemble_preds

subm_data.to_csv("submission.csv", index=False)
subm_data.head()
