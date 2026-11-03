# Lab 4: Data Pipeline and Quality

## Problem definition

The goal of this lab is to build a reproducible machine-learning pipeline using the UCI Adult dataset. The task is to predict whether a person’s recorded annual income is greater than $50,000. The dataset has 48,842 records, 14 input features, and one income label.

The inputs include numerical information such as age, education number, capital gain, capital loss, and hours worked per week. They also include categorical information such as workclass, education, occupation, marital status, race, sex, and native country. The output is a binary income label: `>50K` or `<=50K`.

I evaluate Logistic Regression using accuracy and F1 score. Accuracy shows the overall proportion of correct predictions, while F1 helps assess performance on the higher-income class, which is less common in this dataset. I compare results with and without two derived features: `capital_net` and `hours_per_age`.

The pipeline must avoid data leakage. I split the data into stratified training and test sets before cleaning, fit imputers and numeric clipping limits using training data only, and keep the test rows unchanged until applying the fitted transformations. I also use a fixed random seed so the split and sample checks can be reproduced. Because the dataset contains sensitive demographic features and represents historical census data, the results are for coursework and should not be used to make decisions about real individuals.

## Results

Validation confirmed 48,842 rows and 15 columns. The specified ranges and schema checks passed. Missing values were found in workclass (2,799), occupation (2,809), and native-country (857). There were 52 exact duplicate rows in the full dataset; 24 were in the training split and removed there only. The test split contained 14,653 rows and was kept intact.

The Logistic Regression baseline without derived features achieved 0.8540 accuracy and 0.6601 F1. With `capital_net` and `hours_per_age`, it achieved 0.8578 accuracy and 0.6696 F1.

The label-quality audit found five duplicate-feature groups with conflicting income labels, involving 10 rows. I manually checked 50 sampled rows and found no impossible or inconsistent values.

The processed train and test matrices had shapes 34,165 × 107 and 14,653 × 107. Their SHA256 hash is recorded in `config.yaml` and `results/processed_sha256.txt`.