import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
print("Exploring the Dataset")
df=pd.read_csv("data-preprocessing/iris.csv")
print("First 5 rows of the dataset")
print(df.head())
print("Column and datatypes")
print(df.dtypes)
print("Missing values per column")
print(df.isnull().sum())


print("HANDLING MISSING VALUES")
if df.isnull().sum().sum()==0:
    print("Result There is no missed value in the dataset and so we donot have to replace anything")
print("SPLITING THE DATASET")
target_column=df.columns[-1]
X=df.drop(columns=[target_column])
y=df[target_column]
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=0,stratify=y
)
print(f"Training set shape(X_train) : {X_train.shape}")
print(f"Testing set shape(X_test) : {X_test.shape}")

print("Encoding categorical Variables")
label_encoder=LabelEncoder()
y_train_encoded=label_encoder.fit_transform(y_train)
y_test_encoded=label_encoder.transform(y_test)
print("Target labels successfully encoded into numerical values")

print("SCALING NUMERICAL FEATURES")
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)
print("NUMERICAL FEATURES SCALED SUCCESSFULLY")

print("VERIFYING THE RESULTS")
print(f"X_train scaled shape {X_train_scaled.shape}")
print(f"X_test scaled shape {X_test_scaled.shape}")
print(f"Any missing value left in the X_train {pd.DataFrame(X_train_scaled).isnull().sum().sum()}")
print("Preprossing Completed data is ready for model training")
