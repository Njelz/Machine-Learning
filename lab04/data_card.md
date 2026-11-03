# Data Card: UCI Adult Income

## Source and license

The Adult dataset, also called Census Income, was extracted from the 1994 U.S. Census database. It contains 48,842 records, 14 input features, and an income label. UCI lists it under the CC BY 4.0 license. [1]

## Purpose and contents

This lab uses the data to predict whether recorded income is greater than $50,000. Features include age, education, workclass, occupation, hours per week, race, sex, and native country.

## Quality and processing

The raw data contains missing values in workclass, occupation, and native country. Validation found 52 exact duplicate rows and five groups of duplicate feature rows with conflicting labels. Fifty sampled rows were manually checked against the lab schema and ranges; no impossible or inconsistent values were found.

The data was split 70/30 with stratification and seed 42. Duplicates were removed from training only. Imputation and numeric clipping were fitted on training data only; test rows were kept intact and transformed with the fitted preprocessor.

## Limitations and privacy

This dataset represents a historical U.S. population and should not be treated as representative of people or incomes today. Its income labels and demographic features may reflect historical social biases. Age, race, sex, and country are sensitive attributes, so predictions should not be used to make consequential decisions about individuals.

## Reference

[1]: https://archive.ics.uci.edu/ml/datasets/adult "UCI Adult dataset page"
