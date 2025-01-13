import pdb
import time
from lifelines.utils import concordance_index
import pandas as pd
import numpy as np
from pathlib import Path

from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
from sklearn.model_selection import KFold

import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings(action='ignore')

path = Path('data')
df = pd.read_csv(path/'train.csv')
df = df.drop('ID', axis=1)


def create_multiple_targets(df: pd.DataFrame, time_col: str, event_col: str):
    '''Create multiple target columns using different survival analysis methods'''
    # 1. Kaplan-Meier target
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], event_observed=df[event_col])
    km_target = kmf.survival_function_at_times(df[time_col]).values.flatten()

    # 2. Cox proportional hazards target
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df, duration_col=time_col, event_col=event_col)
    cox_target = cph.predict_partial_hazard(df)

    # 3. Nelson-Aalen target
    naf = NelsonAalenFitter()
    naf.fit(df[time_col], event_observed=df[event_col])
    na_target = -naf.cumulative_hazard_at_times(df[time_col]).values.flatten()

    return km_target, cox_target, na_target


# ================Analysis===================
# drop duplicate
df.drop_duplicates(inplace=True)

# collect categorical and numerical column
cat_cols = df.select_dtypes(include='object').columns.values
num_cols = df.select_dtypes(include=np.number).drop(
    columns=['efs', 'efs_time']).columns.values

# handling missing values
nan_mapping = {'n/a': None, 'na': None, 'nan': None, '-': None}
for col in cat_cols:
    df[col] = df[col].str.strip().str.lower().replace(nan_mapping)

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].apply(lambda col: col.fillna(
    col.mode()[0] if not col.mode().empty else 'unknown'))
df[num_cols] = df[num_cols].astype('float32')

# Factorize categorical vars
for col in df[cat_cols]:
    df[col], _ = pd.factorize(df[col])

# ===============handle target column============
km_target, cox_target, na_target = create_multiple_targets(
    df, time_col='efs_time', event_col='efs')
df['target_km'] = km_target
df['target_cox'] = cox_target
df['target_na'] = na_target
df = df.drop(columns=['efs', 'efs_time'], errors='ignore')

# ================Modeling===================
# load training datga
train_df = pd.read_csv(path/'train.csv')

# handle missing values and categorical columns
train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].median())
train_df[cat_cols] = train_df[cat_cols].apply(lambda col: col.fillna(
    col.mode()[0] if not col.mode().empty else 'unknown'))

for col in cat_cols:
    train_df[col] = train_df[col].astype('category')

train_df['target_km'] = df['target_km']
train_df['target_cox'] = df['target_cox']
train_df['target_na'] = df['target_na']


X = train_df.drop(columns=['efs', 'efs_time', 'ID', 'rituximub',
                  'target_km', 'target_cox', 'target_na'], errors='ignore')
y_dict = {
    'km': df['target_km'],
    'cox': df['target_cox'],
    'na': df['target_na'],
}


# ===========Find Optinal Hyperparamenters==============
# Define base parameters for both models
lgb_params = {
    'objective': 'regression',
    'min_child_samples': 32,
    'num_iterations': 6000,
    'learning_rate': 0.03,
    'num_leaves': 64,
    'max_depth': 8,
    'lambda_l1': 8.0,
    'lambda_l2': 0.1,
    'random_state': 42,
    'verbose': -1
}

ctb_params = {
    'loss_function': 'RMSE',
    'learning_rate': 0.03,
    'random_state': 42,
    'num_trees': 6000,
    'subsample': 0.85,
    'reg_lambda': 8.0,
    'depth': 8
}


def train_model_with_target(X, y, model_type, params, cat_cols, target_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    oof_predictions = np.zeros(len(X))
    c_indices = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        tik = time.time()
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(
                    300, verbose=0), lgb.log_evaluation(0)]

            )
        else:
            model = CatBoostRegressor(
                **params, cat_features=cat_cols, verbose=0)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False
            )
        tok = time.time()
        print(f"time taken to train {tok-tik:.2f}s")
        models.append(model)
        val_preds = model.predict(X_valid)
        oof_predictions[valid_idx] = val_preds

        c_index = concordance_index(y_valid, val_preds)
        c_indices.append(c_index)
        print(
            f"{model_type.upper()} - {target_name} - Fold {fold+1} C-Index {c_index:.4f}")
        break

    mean_c_index = np.mean(c_indices)
    print(f"{model_type.upper()} - {target_name} - Mean C-Index {c_index:.4f}")
    return models, oof_predictions, mean_c_index


results = {}
for target_name, y_target in y_dict.items():
    print(f"\nTraining models for {target_name} target:")

    # Train LightGBM
    lgb_models, lgb_oof, lgb_score = train_model_with_target(
        X, y_target, 'lgb', lgb_params, cat_cols, target_name
    )

    # Traig CatBoost
    print('trainig catboost')
    ctb_models, ctb_oof, ctb_score = train_model_with_target(
        X, y_target, 'ctb', ctb_params, cat_cols, target_name
    )

    results[target_name] = {
        'lgb': {'models': lgb_models, 'oof': lgb_oof, 'score': lgb_score},
        # 'ctb': {'models': ctb_models, 'oof': ctb_oof, 'score': ctb_score}
    }


# ============Inference=============
def make_prediction(test_data, results):
    all_predictions = []
    for target_name, models in results.items():
        lgb_preds = np.mean([model.predict(test_data)
                            for model in models['lgb']['models']], axis=0)
        all_predictions.append(lgb_preds)

        # ctb_preds = np.mean([model.predict(test_data)
        #                     for model in models['ctb']['models']], axis=0)
        # all_predictions.append(ctb_preds)
    final_prediction = np.mean(all_predictions, axis=0)
    return final_prediction


test_df = pd.read_csv(path/'test.csv')
X_test = test_df.drop(columns=['ID', 'rituximub'], errors='ignore')

for col in cat_cols:
    X_test[col] = X_test[col].astype('category')

prediction = make_prediction(X_test, results)
