import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
print("LOADING DATASET")
df=pd.read_csv("linear-regression/house_prices.csv")
print("First rows of the dataset")
print(df.head())
print("Datasets Info and Missing values")
print(df.info())
print(df.isnull().sum())

print("PREPROSSEING AND FEATURE SELECTION")
print("Available columns:",df.columns.tolist())
target_column="price"
feature_column=df.select_dtypes(include=['number']).columns[0]
if target_column not in df.columns:
    target_column=df.columns[-1]
    feature_column=df.columns[0]
print(f"Using feature (X):'{feature_column}' to predict target(y):'{target_column}'")
df_clean=df[[feature_column,target_column]].dropna()
X=df_clean[[feature_column]]
y=df_clean[target_column]

print("SPLITTING THE DATASET")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print("Training set size:{X_train.shape[0]} rows")
print("Testing set size:{X_test.shape[0]} rows")

print("Training the Model")
model=LinearRegression()
model.fit(X_train,y_train)
print("Model training completed")

print("Model Interpretation")
intercept=model.intercept_
coefficient=model.coef_
print(f"Intercept (Y-intercept):{intercept:.2f}")
print(f"Coefficient  (Slope):{coefficient:.2f}")
print(f"Interpretation For every 1 unit increase in '{feature_column}',the predicted'{target_column}' changes by approximately {coefficient:.2f} units ")

print("MODEL EVALUATION")
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"Mean Squared Error (MSE):{mse:.2f}")
print(f"R-squared($R^2$) Score:{r2:.4f}")
print(f"Interpretation: Our model explains about {r2 * 100:.2f}% of the variance in the house prices.")

