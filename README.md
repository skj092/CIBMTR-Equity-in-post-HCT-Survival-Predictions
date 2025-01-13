# CIBMTR-Equity-in-post-HCT-Survival-Predictions


# Solution-1

-------
1. read the train.csv file
2. draop `id` column
3. combine both the target into a single name `target` (ToRead)
4. drop the old targets - `efs` and `efs_time`.
5. drop duplicated rows
6. use single value for missing values in all the categorical columns
7. create a list of categorical and numerical columns
8. Fill missing in numerical columns with median
9. fill missing in categorical columns with model if model is empty use 'unknown'.
10. factorize categorical columns
11. print collelation (ToRead)
----------

12. Again load the train.csv file
13. Take target variable from the processed data
14. Drop the non relevent columns
15. Create X, y for independent and dependent columns.
16. create a list of categorical and numerical columns
17. Define transform for numerical columns - imputing and scaling
18. Define transform for categorical columns - imputing and onehotencoding
19. combine both the transform
20. Define model pipeline with transform and actual model
21. Create 5 fold of data and train the model
22. Find metrics value and make prediction on test dataset.

# Solution-2
## Idea:
- Combine hla columns to make new features
- Create 3 target instead of one
- For every target train `catboost` and `lightgbm`
- Use ensemble model

## Steps:
- create a list of all the columns of hla : Total is 17
- create config which will also include the hyperparameters of models
## Feature Engineering
- define feature engineering class and apply train df file and test df file
    - Feature Engineering: Add appropriate hla columns
    - Fill missing numerical with -1 and categorical with 'unknown'
## Model Development:
- Handle Target for train data:
    1. create one hot encoding for categorical columns
    2. do cross vlidation to create train and valid dataset
    3. drop constant columns if exist
    4. create `CoxPHFitter` target column using 'efs_tie' and 'efs'
    5. Repeate step 1-4 to create 3 more new targets.

## Model Training
- Train `catboost` and `lightgbm` for all the 4 targets.

```python
def recalculate_hla_sums(self, df):
    # Fill null values with 0 for all HLA columns
    hla_columns = [col for col in df.columns if 'hla_match_' in col]
    df[hla_columns] = df[hla_columns].fillna(0)

    # Calculate sums for different HLA groupings
    df['hla_nmdp_6'] = (
        df['hla_match_a_low'] +
        df['hla_match_b_low'] +
        df['hla_match_drb1_high']
    )

    df['hla_low_res_6'] = (
        df['hla_match_a_low'] +
        df['hla_match_b_low'] +
        df['hla_match_drb1_low']
    )

    df['hla_high_res_6'] = (
        df['hla_match_a_high'] +
        df['hla_match_b_high'] +
        df['hla_match_drb1_high']
    )

    df['hla_low_res_8'] = (
        df['hla_match_a_low'] +
        df['hla_match_b_low'] +
        df['hla_match_c_low'] +
        df['hla_match_drb1_low']
    )

    df['hla_high_res_8'] = (
        df['hla_match_a_high'] +
        df['hla_match_b_high'] +
        df['hla_match_c_high'] +
        df['hla_match_drb1_high']
    )

    df['hla_low_res_10'] = (
        df['hla_match_a_low'] +
        df['hla_match_b_low'] +
        df['hla_match_c_low'] +
        df['hla_match_drb1_low'] +
        df['hla_match_dqb1_low']
    )

    df['hla_high_res_10'] = (
        df['hla_match_a_high'] +
        df['hla_match_b_high'] +
        df['hla_match_c_high'] +
        df['hla_match_drb1_high'] +
        df['hla_match_dqb1_high']
    )

    return df
```

-------------------
Overall Stratified C-Index Score for CatBoost: 0.6613
Overall Stratified C-Index Score for LightGBM: 0.6596
Overall Stratified C-Index Score for CatBoost: 0.6751
Overall Stratified C-Index Score for LightGBM: 0.6668

--------------------
Training models for km target:
time taken to train 1.01s
LGB - km - Fold 1 C-Index 0.6392
LGB - km - Mean C-Index 0.6392
trainig catboost
time taken to train 326.30s
CTB - km - Fold 1 C-Index 0.6455
CTB - km - Mean C-Index 0.6455

Training models for cox target:
time taken to train 3.22s
LGB - cox - Fold 1 C-Index 0.9607
LGB - cox - Mean C-Index 0.9607
trainig catboost
time taken to train 322.83s
CTB - cox - Fold 1 C-Index 0.9818
CTB - cox - Mean C-Index 0.9818

Training models for na target:
time taken to train 1.67s
LGB - na - Fold 1 C-Index 0.6406
LGB - na - Mean C-Index 0.6406
trainig catboost
time taken to train 325.98s
CTB - na - Fold 1 C-Index 0.6452
CTB - na - Mean C-Index 0.6452

