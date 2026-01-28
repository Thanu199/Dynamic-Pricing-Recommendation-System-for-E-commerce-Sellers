import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data from ZIP file

file_path = r"C:\Users\Thanusri\dp_project\data\Amazon Sale Report.csv.zip"

print("Loading dataset...")

# Pandas can directly read CSV from zip
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Drop Unwanted Columns

drop_columns = [
    "index",
    "Order ID",
    "promotion-ids",
    "ship-postal-code",
    "ship-country",
    "Unnamed: 22"
]

for col in drop_columns:
    if col in df.columns:
        df = df.drop(columns=col)

print("\nAfter dropping unwanted columns:")
print("Shape:", df.shape)

# 3. Handle Date Column

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["order_year"] = df["Date"].dt.year
    df["order_month"] = df["Date"].dt.month
    df["order_day"] = df["Date"].dt.day
    df = df.drop(columns=["Date"])

print("\nAfter date processing:")
print(df[["order_year", "order_month", "order_day"]].head())

# 4. Drop rows with missing target

if "Amount" not in df.columns:
    raise ValueError("Target column 'Amount' not found in dataset.")

df = df.dropna(subset=["Amount"])

print("\nAfter dropping rows with missing Amount:")
print("Shape:", df.shape)

# 5. Fill Missing Values

# Numerical columns → median
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns → "Unknown"
categorical_cols = df.select_dtypes(include=["object", "bool"]).columns
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

print("\nMissing values handled.")

# 6. Define Features and Target

target = "Amount"
X = df.drop(columns=[target])
y = df[target]

print("\nFinal feature columns:")
print(X.columns.tolist())
print("Number of features:", X.shape[1])

# 7. Build Preprocessing Pipeline

categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# 8. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# 9. Build and Train Model (XGBoost)

from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ]
)

print("\nTraining XGBoost model...")

pipeline.fit(X_train, y_train)

print("XGBoost training completed.")

# 10. Evaluate Model

y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

y_train_pred = pipeline.predict(X_train)
rscore = r2_score(y_train, y_train_pred)

print("\nModel Performance:")
print("MSE:", mse)
print("R2 Score:", r2)
print("r2 score for train : ",rscore)

# 11. Save Model

model_path = "pricing_model.pkl"
joblib.dump(pipeline, model_path)
print("\nModel saved as:", model_path)