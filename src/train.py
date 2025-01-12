from lifelines.utils import concordance_index
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import code
import pandas as pd
import numpy as np
from pathlib import Path

from lifelines import KaplanMeierFitter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold

path = Path('data')
df = pd.read_csv(path/'train.csv')
df = df.drop('ID', axis=1)


def kaplan(df: pd.DataFrame, time_col: str, event_col: str):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], event_observed=df[event_col])
    return kmf.survival_function_at_times(df[time_col]).values.flatten()


# ================Analysis===================

# handle target column
df['target'] = kaplan(df, time_col='efs_time', event_col='efs')
df = df.drop(columns=['efs', 'efs_time'], errors='ignore')

# drop duplicate
df.drop_duplicates(inplace=True)

# collect categorical and numerical column
cat_cols = df.select_dtypes(include='object').columns.values
num_cols = df.select_dtypes(include=np.number).drop(
    columns='target').columns.values

# handling missing values
nan_mapping = {'n/a': None, 'na': None, 'nan': None, '-': None}
for col in cat_cols:
    df[col] = df[col].str.strip().str.lower().replace(nan_mapping)

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].apply(lambda col: col.fillna(
    col.mode()[0] if not col.mode().empty else 'unknown'))

# Factorize categorical vars
for col in df[cat_cols]:
    df[col], _ = pd.factorize(df[col])

# correlation
correlatio_map = {col: df[col].corr(df['target']) for col in df.columns}

# ================Modeling===================
train_df = pd.read_csv(path/'train.csv')
train_df['target'] = df['target']

X = train_df.drop(columns=['efs', 'efs_time', 'ID',
                  'target', 'rituximub'], errors='ignore')
y = df['target']

# transform for numerical columns
num_tfms = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_tfms = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_tfms, num_cols),
        ('cat', cat_tfms, cat_cols)
    ]
)

# create model pipeline
model = Pipeline(steps=[
                 ('preprocessor', preprocessor),
                 ('classifier', GradientBoostingRegressor(random_state=42))
                 ])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
c_indices = []
for train_index, val_index in kf.split(X):
    x_train, x_valid = X.iloc[train_index], X.iloc[val_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[val_index]

    model.fit(x_train, y_train)
    val_probs = model.predict(x_valid)
    c_index = concordance_index(y_valid, val_probs)
    c_indices.append(c_index)
    print(f"stratified C-Index : {c_index:.4f}")

print(f"mean stratified C-Index : {np.mean(c_indices):.4f}")
