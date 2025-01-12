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
