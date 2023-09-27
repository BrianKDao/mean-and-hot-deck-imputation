# mean-and-hot-deck-imputation
This assignment asks you to implement and evaluate two popular algorithms for the imputation of missing values using two provided datasets. You will evaluate and compare their runtime and the quality of the imputed values by comparing them with the corresponding values in the “complete” dataset.

Assignment 2 for cmsc435-Fall-2023

Done by Brian Dao

Data Sets
- dataset_complete.csv file is the complete dataset. It includes 10 features and 8795 objects.
- dataset_missing01.csv and dataset_missing10.csv files include the same dataset with 1% and 10% of
  missing values, respectively.

I chose to write this in python so I was restricted to these rules:

- Your code should be in one source code (.py) file. You may define any number of classes and
functions, but everything must be included in that file.
- You are only allowed to use NumPy (https://www.numpy.org/) and pandas
(https://pandas.pydata.org) as imported libraries. They may help you with reading the csv files
and working with the data.
- Your program will be tested on Python 3.11 with the latest (as of today) versions of numpy
(1.25) and pandas (2.1.0) installed.
- This python file must successfully run on the above python environment and produce the above-
mentioned outputs with the required precision. It should be run by executing the below command
in the location where three input csv files are located.
    python3 a2.py
For this, you need to make sure you read the input csv files from the current working directory.