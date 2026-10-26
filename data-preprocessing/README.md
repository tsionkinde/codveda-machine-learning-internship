Project Overview
This project demonstrates the main steps used to prepare a dataset for
machine learning with Python, pandas, and scikit-learn.

The goal is to inspect a dataset, handle missing values where necessary,
encode categorical features, scale numerical features when appropriate,
and split the data into training and testing sets.
Dataset
Planned dataset: 1) iris.csv

Place the CSV file in the project folder. Before preprocessing, inspect
its columns, data types, missing values, and target column. The exact
preprocessing steps will depend on what the dataset contains.
Tools and Libraries

- Python
- pandas
- scikit-learn
- Jupyter Notebook or VS Code
  Install the required libraries:
  pip install pandas scikit-learn jupyter
  Project Structure
  task-1-data-preprocessing/
  ├── README.md
  ├── iris.csv
  └── task1_preprocessing.ipynb
  You can use a Python script named task1_preprocessing.py instead of a
  notebook if you prefer.
  Tasks

1. Explore the dataset

- Load the CSV file with pandas.
- Display the first rows.
- Inspect column names and data types.
- Count missing values in each column.

2. Handle missing values

- Use the mean or median for appropriate numerical columns.
- Use the most frequent category for categorical columns when
  suitable.
- Drop rows or columns only when there is a good reason.
- If the dataset has no missing values, report that rather than making
  unnecessary changes.

3. Encode categorical variables

- Use one-hot encoding for nominal categories when appropriate.
- Use label/ordinal encoding only when its meaning suits the data.
- If the dataset has no categorical input features, explain that
  encoding is not needed.

4. Scale numerical features

- Use StandardScaler for standardization or MinMaxScaler for
  normalization when appropriate.
- Fit preprocessing steps on the training set only to avoid data
  leakage.

5. Split the dataset

- Separate input features (X) from the target (y).
- Split the data into training and testing sets, commonly 80% training
  and 20% testing.
- Use a fixed random_state for reproducibility.
- Use stratification for classification when appropriate and supported
  by the data.

6. Verify the results

- Check the shapes of the training and testing sets.
- Confirm that missing values have been handled.
- Confirm that categorical features are encoded and numerical features
  are scaled as intended.
  Important Practice
  Split the dataset before fitting imputers, encoders, or scalers. Fit
  these preprocessing tools using the training data, then use the fitted
  tools to transform the test data. This helps prevent data leakage.
  Expected Outcome
  At the end of this task, the dataset should be prepared for a machine
  learning model, with a clear explanation of each preprocessing decision.
  How to Run

1. Put iris.csv in the project folder.
2. Open task1_preprocessing.ipynb in Jupyter or VS Code.
3. Run the notebook cells from top to bottom.
4. Record the findings and any preprocessing steps that were
   unnecessary for this dataset.
