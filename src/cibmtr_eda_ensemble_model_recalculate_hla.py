# -*- coding: utf-8 -*-
"""cibmtr-eda-ensemble-model-recalculate-hla.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nT0nX4x64CDPEspc-8WrD6D_VeQaM2uW

<p style="background-color: rgb(247, 230, 202); font-size: 300%; text-align: center; border-radius: 40px 40px; color: rgb(162, 87, 79); font-weight: bold; font-family: 'Roboto'; border: 4px solid rgb(162, 87, 79);">Introduction</p>

Addition in this notebook is the recalculation of HLA sums which are often missing. Since the sum is full of missing values, it was interesting to modify the values by recalculating them based on the data dictionary explanations and see the results.
"""

# HLA : Human Leukocyte Antigen matching levels.
# Homozygous chromosomes have the same allele at a given locus (fixed position on a chromosome where a particular gene is located).
# Those that have different alleles at a given locus are called heterozygous.
# Values Explanation
# 0 - No Match: Neither of the donor's two HLA antigens/alleles matches the recipient's HLA.
# This indicates a complete mismatch at the locus, which increases the risk of complications like graft rejection or graft-versus-host disease (GVHD).
# 1 - Partial Match: One of the donor's HLA antigens/alleles matches one of the recipient's.
# This represents a half-match (heterozygous compatibility) at the locus. It's better than a full mismatch but still carries a moderate risk of immune complications.
# 2 - Full Match: Both of the donor's HLA match both of the recipient's HLA.
# This is the optimal scenario, indicating full compatibility at the locus and minimizing the risk of immune complications.

# High VS Low resolution

# High-Resolution Typing: Identifies specific alleles (e.g., HLA-A*02:01).
# Provides the most precise match and is essential for unrelated donor transplants.

# Low-Resolution Typing: Identifies broader antigen groups (e.g., HLA-A2).
# May suffice for related donor transplants where genetic similarity is inherently higher.

from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from scipy.stats import rankdata
from metric import score
import lightgbm as lgb
from lifelines import NelsonAalenFitter
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
import warnings
HLA_COLUMNS = [
    # MHC class I molecules are one of two primary classes of major histocompatibility complex (MHC) molecules and are found on the cell surface of all nucleated cells.
    # In humans, the HLAs corresponding to MHC class I are HLA-A, HLA-B, and HLA-C.
    'hla_match_a_low', 'hla_match_a_high',
    'hla_match_b_low', 'hla_match_b_high',
    'hla_match_c_low', 'hla_match_c_high',

    # MHC Class II molecules are a class of major histocompatibility complex (MHC) molecules normally found only on professional antigen-presenting cells
    # such as dendritic cells, macrophages, some endothelial cells, thymic epithelial cells, and B cells.
    # Antigens presented by MHC class II molecules are exogenous, originating from extracellular proteins rather than cytosolic and endogenous sources like
    # those presented by MHC class I.
    # HLAs corresponding to MHC class II are HLA-DP, HLA-DM, HLA-DOA, HLA-DOB, HLA-DQ, and HLA-DR.
    # In this competition, we only have HLA-DR and HLA-DQ

    # Visit https://en.wikipedia.org/wiki/HLA-DQB1

    'hla_match_dqb1_low', 'hla_match_dqb1_high',
    'hla_match_drb1_low', 'hla_match_drb1_high',

    # Combination of matches : sum of matches between multiple categories

    # Matching at HLA-A(low), -B(low), -DRB1(high)
    'hla_nmdp_6',
    # Matching at HLA-A,-B,-DRB1 (low or high)
    'hla_low_res_6', 'hla_high_res_6',
    # Matching at HLA-A, -B, -C, -DRB1 (low or high)
    'hla_low_res_8', 'hla_high_res_8',
    # Matching at HLA-A, -B, -C, -DRB1, -DQB1 (low or high)
    'hla_low_res_10', 'hla_high_res_10'
]

"""<div style="background-color: rgb(247, 230, 202); border: 4px solid rgb(162, 87, 79); border-radius: 40px; padding: 20px; font-family: 'Roboto'; color: rgb(162, 87, 79); text-align: left; font-size: 120%;">
    <ul style="list-style-type: square; padding-left: 20px;">
        <li>Missing values are replaced with:
            <ul style="list-style-type: circle; margin-top: 10px; margin-bottom: 10px;">
                <li>-1 for numeric columns</li>
                <li>Unknown for categorical columns</li>
            </ul>
        </li>
        <li style="margin-top: 10px;">
            LightGBM and CatBoost are trained on 3 different targets, estimated from the survival models:
            <ul style="list-style-type: circle; margin-top: 10px; margin-bottom: 10px;">
                <li>Cox</li>
                <li>Kaplan-Meier</li>
                <li>Nelson-Aalen</li>
            </ul>
        </li>
        <li style="margin-top: 10px;">Two additional CatBoost model are trained, with Cox loss function.</li>
        <li style="margin-top: 10px;">As per <a href="https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/553061" style="color: #A2574F; text-decoration: underline;">this</a> discussion post, the target is consisted of the Out-of-Fold predictions of the survival models on the validation folds to prevent target leakage.</li>
        <li style="margin-top: 10px;">
            The ensemble prediction for each sample is computed as:
            <ul style="list-style-type: circle; margin-top: 10px; margin-bottom: 10px;">
                <p style="margin-top: 10px; font-size: 110%; color: #A2574F; font-family: 'Roboto'; text-align: left;">
                    $ \text{preds}_{\text{ensemble}} = \sum_{i=1}^{n} w_i \cdot \text{rankdata}(\text{preds}_i) $
                </p>
                where $n$ is the number of models, $w_i$ is the weight assigned to the $i$-th model, and $\text{rankdata}(\text{preds}_i)$ is the rank of predictions from the $i$-th model.
            </ul>
        </li>
        <li style="margin-top: 10px;">Last but not least, since the competition metric evaluates only the order of predictions and not their magnitude, the model weights are not required to sum to 1, nor should the predictions fall within a predefined range.</li>
    </ul>
</div>

<p style="background-color: rgb(247, 230, 202); font-size: 300%; text-align: center; border-radius: 40px 40px; color: rgb(162, 87, 79); font-weight: bold; font-family: 'Roboto'; border: 4px solid rgb(162, 87, 79);">Install Libraries</p>
"""

# !pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
# !pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
# !pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
# !pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
# !pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl
#
# """<p style="background-color: rgb(247, 230, 202); font-size: 300%; text-align: center; border-radius: 40px 40px; color: rgb(162, 87, 79); font-weight: bold; font-family: 'Roboto'; border: 4px solid rgb(162, 87, 79);">Imports</p>"""

warnings.filterwarnings('ignore')


pio.renderers.default = 'iframe'

pd.options.display.max_columns = None


"""<p style="background-color: rgb(247, 230, 202); font-size: 300%; text-align: center; border-radius: 40px 40px; color: rgb(162, 87, 79); font-weight: bold; font-family: 'Roboto'; border: 4px solid rgb(162, 87, 79);">Configuration</p>"""


class CFG:

    path = Path('data')
    train_path = path/'train.csv'
    test_path = path/'test.csv'
    subm_path = path/'sample_submission.csv'

    color = '#A2574F'

    batch_size = 32768
    early_stop = 300
    penalizer = 0.01
    n_splits = 5

    weights = [1.0, 1.0, 8.0, 4.0, 8.0, 4.0, 6.0, 6.0]

    ctb_params = {
        'loss_function': 'RMSE',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'subsample': 0.85,
        'reg_lambda': 8.0,
        'depth': 8
    }

    lgb_params = {
        'objective': 'regression',
        'min_child_samples': 32,
        'num_iterations': 6000,
        'learning_rate': 0.03,
        'extra_trees': True,
        'reg_lambda': 8.0,
        'reg_alpha': 0.1,
        'num_leaves': 64,
        'metric': 'rmse',
        'max_depth': 8,
        'device': 'cpu',
        'max_bin': 128,
        'verbose': -1,
        'seed': 42
    }

    # Parameters for the first CatBoost model with Cox loss function
    cox1_params = {
        'grow_policy': 'Depthwise',
        'min_child_samples': 8,
        'loss_function': 'Cox',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'subsample': 0.85,
        'reg_lambda': 8.0,
        'depth': 8
    }

    # Parameters for the second CatBoost model with Cox loss function
    cox2_params = {
        'grow_policy': 'Lossguide',
        'loss_function': 'Cox',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'subsample': 0.85,
        'reg_lambda': 8.0,
        'num_leaves': 32,
        'depth': 8
    }


"""<p style="background-color: rgb(247, 230, 202); font-size: 300%; text-align: center; border-radius: 40px 40px; color: rgb(162, 87, 79); font-weight: bold; font-family: 'Roboto'; border: 4px solid rgb(162, 87, 79);">Feature Engineering</p>"""


class FE:

    def __init__(self, batch_size):
        self._batch_size = batch_size

    def load_data(self, path):

        return pl.read_csv(path, batch_size=self._batch_size)

    def recalculate_hla_sums(self, df):

        df = df.with_columns(
            (pl.col("hla_match_a_low").fill_null(0) + pl.col("hla_match_b_low").fill_null(0) +
             pl.col("hla_match_drb1_high").fill_null(0)).alias("hla_nmdp_6"),

            (pl.col("hla_match_a_low").fill_null(0) + pl.col("hla_match_b_low").fill_null(0) +
             pl.col("hla_match_drb1_low").fill_null(0)).alias("hla_low_res_6"),

            (pl.col("hla_match_a_high").fill_null(0) + pl.col("hla_match_b_high").fill_null(0) +
             pl.col("hla_match_drb1_high").fill_null(0)).alias("hla_high_res_6"),

            (pl.col("hla_match_a_low").fill_null(0) + pl.col("hla_match_b_low").fill_null(0) +
             pl.col("hla_match_c_low").fill_null(0) +
             pl.col("hla_match_drb1_low").fill_null(0)
             ).alias("hla_low_res_8"),

            (pl.col("hla_match_a_high").fill_null(0) + pl.col("hla_match_b_high").fill_null(0) +
             pl.col("hla_match_c_high").fill_null(0) +
             pl.col("hla_match_drb1_high").fill_null(0)
             ).alias("hla_high_res_8"),

            (pl.col("hla_match_a_low").fill_null(0) + pl.col("hla_match_b_low").fill_null(0) +
             pl.col("hla_match_c_low").fill_null(0) + pl.col("hla_match_drb1_low").fill_null(0) +
             pl.col("hla_match_dqb1_low").fill_null(0)).alias("hla_low_res_10"),

            (pl.col("hla_match_a_high").fill_null(0) + pl.col("hla_match_b_high").fill_null(0) +
             pl.col("hla_match_c_high").fill_null(0) + pl.col("hla_match_drb1_high").fill_null(0) +
             pl.col("hla_match_dqb1_high").fill_null(0)).alias("hla_high_res_10"),
        )

        return df

    def cast_datatypes(self, df):

        num_cols = [
            'hla_high_res_8',
            'hla_low_res_8',
            'hla_high_res_6',
            'hla_low_res_6',
            'hla_high_res_10',
            'hla_low_res_10',
            'hla_match_dqb1_high',
            'hla_match_dqb1_low',
            'hla_match_drb1_high',
            'hla_match_drb1_low',
            'hla_nmdp_6',
            'year_hct',
            'hla_match_a_high',
            'hla_match_a_low',
            'hla_match_b_high',
            'hla_match_b_low',
            'hla_match_c_high',
            'hla_match_c_low',
            'donor_age',
            'age_at_hct',
            'comorbidity_score',
            'karnofsky_score',
            'efs',
            'efs_time'
        ]

        for col in df.columns:

            if col in num_cols:
                df = df.with_columns(
                    pl.col(col).fill_null(-1).cast(pl.Float32))

            else:
                df = df.with_columns(
                    pl.col(col).fill_null('Unknown').cast(pl.String))

        return df.with_columns(pl.col('ID').cast(pl.Int32))

    def info(self, df):

        print(f'\nShape of dataframe: {df.shape}')

        mem = df.memory_usage().sum() / 1024**2
        print('Memory usage: {:.2f} MB\n'.format(mem))

        # display(df.head())

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

"""<p style="background-color: rgb(247, 230, 202); font-size: 300%; text-align: center; border-radius: 40px 40px; color: rgb(162, 87, 79); font-weight: bold; font-family: 'Roboto'; border: 4px solid rgb(162, 87, 79);">Model Development</p>"""


class EDA:

    def __init__(self, color, data):
        self._color = color
        self.data = data

    def _template(self, fig, title):

        fig.update_layout(
            title=title,
            title_x=0.5,
            plot_bgcolor='rgba(247, 230, 202, 1)',
            paper_bgcolor='rgba(247, 230, 202, 1)',
            font=dict(color=self._color),
            margin=dict(l=72, r=72, t=72, b=72),
            height=720
        )

        return fig

    def distribution_plot(self, col, title):

        fig = px.histogram(
            self.data,
            x=col,
            nbins=100,
            color_discrete_sequence=[self._color]
        )

        fig.update_layout(
            xaxis_title='Values',
            yaxis_title='Count',
            bargap=0.1,
            xaxis=dict(gridcolor='grey'),
            yaxis=dict(gridcolor='grey', zerolinecolor='grey')
        )

        fig.update_traces(hovertemplate='Value: %{x:.2f}<br>Count: %{y:,}')

        fig = self._template(fig, f'{title}')
        # fig.show()

    def _plot_cv(self, scores, title, metric='Stratified C-Index'):

        fold_scores = [round(score, 3) for score in scores]
        mean_score = round(np.mean(scores), 3)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(1, len(fold_scores) + 1)),
            y=fold_scores,
            mode='markers',
            name='Fold Scores',
            marker=dict(size=27, color=self._color, symbol='diamond'),
            text=[f'{score:.3f}' for score in fold_scores],
            hovertemplate='Fold %{x}: %{text}<extra></extra>',
            hoverlabel=dict(font=dict(size=18))
        ))

        fig.add_trace(go.Scatter(
            x=[1, len(fold_scores)],
            y=[mean_score, mean_score],
            mode='lines',
            name=f'Mean: {mean_score:.3f}',
            line=dict(dash='dash', color='#B22222'),
            hoverinfo='none'
        ))

        fig.update_layout(
            title=f'{title} | Cross-validation Mean {metric} Score: {mean_score}',
            xaxis_title='Fold',
            yaxis_title=f'{metric} Score',
            plot_bgcolor='rgba(247, 230, 202, 1)',
            paper_bgcolor='rgba(247, 230, 202, 1)',
            font=dict(color=self._color),
            xaxis=dict(
                gridcolor='grey',
                tickmode='linear',
                tick0=1,
                dtick=1,
                range=[0.5, len(fold_scores) + 0.5],
                zerolinecolor='grey'
            ),
            yaxis=dict(
                gridcolor='grey',
                zerolinecolor='grey'
            )
        )

        # fig.show()


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

        y_true = self.data[['ID', 'efs', 'efs_time', 'race_group']].copy()
        y_pred = self.data[['ID']].copy()

        y_pred['prediction'] = preds

        c_index_score = score(y_true.copy(), y_pred.copy(), 'ID')
        print(
            f'Overall Stratified C-Index Score for {title}: {c_index_score:.4f}')

    def create_target1(self):
        '''
        Inside the CV loop, constant columns are dropped if they exist in a fold. Otherwise, the code produces error:

        delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation:
        https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
        '''

        cv, oof_preds = self._prepare_cv()

        # Apply one hot encoding to categorical columns
        data = pd.get_dummies(self.data, columns=self.cat_cols,
                              drop_first=True).drop('ID', axis=1)

        for train_index, valid_index in cv.split(data):

            train_data = data.iloc[train_index]
            valid_data = data.iloc[valid_index]

            # Drop constant columns if they exist
            train_data = train_data.loc[:, train_data.nunique() > 1]
            valid_data = valid_data[train_data.columns]

            cph = CoxPHFitter(penalizer=self._penalizer)
            cph.fit(train_data, duration_col='efs_time', event_col='efs')

            oof_preds[valid_index] = cph.predict_partial_hazard(valid_data)

        self.data['target1'] = oof_preds
        self.validate_model(oof_preds, 'Cox')

        return self.data

    def create_target2(self):

        cv, oof_preds = self._prepare_cv()

        for train_index, valid_index in cv.split(self.data):

            train_data = self.data.iloc[train_index]
            valid_data = self.data.iloc[valid_index]

            kmf = KaplanMeierFitter()
            kmf.fit(durations=train_data['efs_time'],
                    event_observed=train_data['efs'])

            oof_preds[valid_index] = kmf.survival_function_at_times(
                valid_data['efs_time']).values

        self.data['target2'] = oof_preds
        self.validate_model(oof_preds, 'Kaplan-Meier')

        return self.data

    def create_target3(self):

        cv, oof_preds = self._prepare_cv()

        for train_index, valid_index in cv.split(self.data):

            train_data = self.data.iloc[train_index]
            valid_data = self.data.iloc[valid_index]

            naf = NelsonAalenFitter()
            naf.fit(durations=train_data['efs_time'],
                    event_observed=train_data['efs'])

            oof_preds[valid_index] = - \
                naf.cumulative_hazard_at_times(valid_data['efs_time']).values

        self.data['target3'] = oof_preds
        self.validate_model(oof_preds, 'Nelson-Aalen')

        return self.data

    def create_target4(self):

        self.data['target4'] = self.data.efs_time.copy()
        self.data.loc[self.data.efs == 0, 'target4'] *= -1

        return self.data


class MD:

    def __init__(self, color, data, cat_cols, early_stop, penalizer, n_splits):

        self.eda = EDA(color, data)
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

        for col in self.cat_cols:
            self.data[col] = self.data[col].astype('category')

        X = self.data.drop(['ID', 'efs', 'efs_time', 'target1',
                           'target2', 'target3', 'target4'], axis=1)
        y = self.data[target]

        models, fold_scores = [], []

        cv, oof_preds = self.targets._prepare_cv()

        for fold, (train_index, valid_index) in enumerate(cv.split(X, y)):

            X_train = X.iloc[train_index]
            X_valid = X.iloc[valid_index]

            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]

            if title.startswith('LightGBM'):

                model = lgb.LGBMRegressor(**params)

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='rmse',
                    callbacks=[lgb.early_stopping(
                        self._early_stop, verbose=0), lgb.log_evaluation(0)]
                )

            elif title.startswith('CatBoost'):

                model = CatBoostRegressor(
                    **params, verbose=0, cat_features=self.cat_cols)

                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_valid, y_valid),
                    early_stopping_rounds=self._early_stop,
                    verbose=0
                )

            models.append(model)

            oof_preds[valid_index] = model.predict(X_valid)

            y_true_fold = self.data.iloc[valid_index][[
                'ID', 'efs', 'efs_time', 'race_group']].copy()
            y_pred_fold = self.data.iloc[valid_index][['ID']].copy()

            y_pred_fold['prediction'] = oof_preds[valid_index]

            fold_score = score(y_true_fold, y_pred_fold, 'ID')
            fold_scores.append(fold_score)

        self.eda._plot_cv(fold_scores, title)
        self.targets.validate_model(oof_preds, title)

        return models, oof_preds

    def infer_model(self, data, models):

        data = data.drop(['ID'], axis=1)

        for col in self.cat_cols:
            data[col] = data[col].astype('category')

        return np.mean([model.predict(data) for model in models], axis=0)


md = MD(CFG.color, train_data, cat_cols,
        CFG.early_stop, CFG.penalizer, CFG.n_splits)

train_data = md.create_targets()

md.eda.distribution_plot('target1', 'Cox Target')

md.eda.distribution_plot('target2', 'Kaplan-Meier Target')

md.eda.distribution_plot('target3', 'Nelson-Aalen Target')

md.eda.distribution_plot('target4', 'Target for Cox-Loss Models')

fe.info(train_data)

"""<p style="background-color: rgb(247, 230, 202); font-size: 300%; text-align: center; border-radius: 40px 40px; color: rgb(162, 87, 79); font-weight: bold; font-family: 'Roboto'; border: 4px solid rgb(162, 87, 79);">Models with Cox Target</p>"""

ctb1_models, ctb1_oof_preds = md.train_model(
    CFG.ctb_params, target='target1', title='CatBoost')

lgb1_models, lgb1_oof_preds = md.train_model(
    CFG.lgb_params, target='target1', title='LightGBM')

ctb1_preds = md.infer_model(test_data, ctb1_models)

lgb1_preds = md.infer_model(test_data, lgb1_models)

"""<p style="background-color: rgb(247, 230, 202); font-size: 300%; text-align: center; border-radius: 40px 40px; color: rgb(162, 87, 79); font-weight: bold; font-family: 'Roboto'; border: 4px solid rgb(162, 87, 79);">Models with Kaplan-Meier Target</p>"""

ctb2_models, ctb2_oof_preds = md.train_model(
    CFG.ctb_params, target='target2', title='CatBoost')

lgb2_models, lgb2_oof_preds = md.train_model(
    CFG.lgb_params, target='target2', title='LightGBM')

ctb2_preds = md.infer_model(test_data, ctb2_models)

lgb2_preds = md.infer_model(test_data, lgb2_models)

"""<p style="background-color: rgb(247, 230, 202); font-size: 300%; text-align: center; border-radius: 40px 40px; color: rgb(162, 87, 79); font-weight: bold; font-family: 'Roboto'; border: 4px solid rgb(162, 87, 79);">Models with Nelson-Aalen Target</p>"""

ctb3_models, ctb3_oof_preds = md.train_model(
    CFG.ctb_params, target='target3', title='CatBoost')

lgb3_models, lgb3_oof_preds = md.train_model(
    CFG.lgb_params, target='target3', title='LightGBM')

ctb3_preds = md.infer_model(test_data, ctb3_models)

lgb3_preds = md.infer_model(test_data, lgb3_models)

"""<p style="background-color: rgb(247, 230, 202); font-size: 300%; text-align: center; border-radius: 40px 40px; color: rgb(162, 87, 79); font-weight: bold; font-family: 'Roboto'; border: 4px solid rgb(162, 87, 79);">Cox-Loss Models</p>"""

cox1_models, cox1_oof_preds = md.train_model(
    CFG.cox1_params, target='target4', title='CatBoost')

cox2_models, cox2_oof_preds = md.train_model(
    CFG.cox2_params, target='target4', title='CatBoost')

cox1_preds = md.infer_model(test_data, cox1_models)

cox2_preds = md.infer_model(test_data, cox2_models)

"""<p style="background-color: rgb(247, 230, 202); font-size: 300%; text-align: center; border-radius: 40px 40px; color: rgb(162, 87, 79); font-weight: bold; font-family: 'Roboto'; border: 4px solid rgb(162, 87, 79);">Ensemble Model</p>

<div style="background-color: rgb(247, 230, 202); border: 4px solid rgb(162, 87, 79); border-radius: 40px; padding: 20px; font-family: 'Roboto'; color: rgb(162, 87, 79); text-align: left; font-size: 140%;">
    <b>Calculate C-Index score for Ensemble model using Out-of-Fold (OOF) predictions.</b>
</div>
"""

oof_preds = [
    ctb1_oof_preds,
    lgb1_oof_preds,
    ctb2_oof_preds,
    lgb2_oof_preds,
    ctb3_oof_preds,
    lgb3_oof_preds,
    cox1_oof_preds,
    cox2_oof_preds
]

ranked_oof_preds = np.array([rankdata(p) for p in oof_preds])

ensemble_oof_preds = np.dot(CFG.weights, ranked_oof_preds)

md.targets.validate_model(ensemble_oof_preds, 'Ensemble Model')

"""<div style="background-color: rgb(247, 230, 202); border: 4px solid rgb(162, 87, 79); border-radius: 40px; padding: 20px; font-family: 'Roboto'; color: rgb(162, 87, 79); text-align: left; font-size: 140%;">
    <b>Ensemble predictions for the test data.</b>
</div>
"""

preds = [
    ctb1_preds,
    lgb1_preds,
    ctb2_preds,
    lgb2_preds,
    ctb3_preds,
    lgb3_preds,
    cox1_preds,
    cox2_preds
]

ranked_preds = np.array([rankdata(p) for p in preds])

ensemble_preds = np.dot(CFG.weights, ranked_preds)

subm_data = pd.read_csv(CFG.subm_path)
subm_data['prediction'] = ensemble_preds

subm_data.to_csv('submission.csv', index=False)
# display(subm_data.head())