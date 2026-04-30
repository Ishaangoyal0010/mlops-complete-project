import pandas as pd
import numpy as np
import os
import pickle
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# 1. Load Data

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
file_path = os.path.join(BASE_DIR, "data", "cleaned_quikr_car.csv")
df = pd.read_csv(file_path)


# 2. Feature Engineering

current_year = 2026
df["car_age"] = current_year - df["year"]
df.drop("year", axis=1, inplace=True)


# 3. Outlier Removal (IQR method)

def remove_outliers(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[col] >= lower) & (data[col] <= upper)]

df = remove_outliers(df, "Price")
df = remove_outliers(df, "kms_driven")


# 4. Split features and target

X = df.drop("Price", axis=1)
y = df["Price"]


# 5. Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Preprocessing (OHE)

categorical_features = ["name", "company", "fuel_type"]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
    remainder="passthrough"
)


# 7. Models

lr_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

# ----------------(2)
# 8. MLflow Setup

mlflow.set_experiment("quikr_car_price_prediction")


# 9. Cross Validation

lr_cv = cross_val_score(lr_model, X_train, y_train, cv=5, scoring="r2")
rf_cv = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="r2")

print("Linear Regression CV R2:", round(lr_cv.mean(), 4))
print("Random Forest CV R2:    ", round(rf_cv.mean(), 4))


# 10. Hyperparameter Tuning (Random Forest)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth":    [None, 10, 20],
    "model__max_features": ["sqrt", "log2"],
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest RF Params:", grid_search.best_params_)
print("Best RF CV R2: ", round(grid_search.best_score_, 4))

rf_model = grid_search.best_estimator_


# 11. Train Linear Regression

lr_model.fit(X_train, y_train)


# 12. Evaluate on Test Set

def evaluate(name, y_true, y_pred):
    print(f"\n{name}")
    print("R2 Score:", round(r2_score(y_true, y_pred), 4))
    print("MAE:     ", round(mean_absolute_error(y_true, y_pred), 2))

lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

evaluate("Linear Regression", y_test, lr_pred)
evaluate("Random Forest",     y_test, rf_pred)

lr_score = r2_score(y_test, lr_pred)
rf_score = r2_score(y_test, rf_pred)
lr_mae   = mean_absolute_error(y_test, lr_pred)
rf_mae   = mean_absolute_error(y_test, rf_pred)

# ---------------(2)
# 13. Log to MLflow


# LR

with mlflow.start_run(run_name="Linear Regression"):
    mlflow.log_metric("cv_r2_mean", round(lr_cv.mean(), 4))
    mlflow.log_metric("cv_r2_std",  round(lr_cv.std(),  4))
    mlflow.log_metric("test_r2",    round(lr_score, 4))
    mlflow.log_metric("test_mae",   round(lr_mae,   2))
    mlflow.sklearn.log_model(lr_model, "model")
    print("\n[MLflow] Linear Regression run logged.")

# RF

with mlflow.start_run(run_name="Random Forest"):
    best_params = grid_search.best_params_
    mlflow.log_param("n_estimators", best_params["model__n_estimators"])
    mlflow.log_param("max_depth",    best_params["model__max_depth"])
    mlflow.log_param("max_features", best_params["model__max_features"])

    mlflow.log_metric("cv_r2_mean", round(rf_cv.mean(), 4))
    mlflow.log_metric("cv_r2_std",  round(rf_cv.std(),  4))
    mlflow.log_metric("test_r2",    round(rf_score, 4))
    mlflow.log_metric("test_mae",   round(rf_mae,   2))
    mlflow.sklearn.log_model(rf_model, "model")
    print("[MLflow] Random Forest run logged.")


# 14. Save Best Model

if rf_score > lr_score:
    best_model      = rf_model
    best_model_name = "Random Forest"
else:
    best_model      = lr_model
    best_model_name = "Linear Regression"

artifact_dir = os.path.join(BASE_DIR, "artifacts")
os.makedirs(artifact_dir, exist_ok=True)

model_path = os.path.join(artifact_dir, "best_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

print(f"\nBest Model: {best_model_name}")
print(f"Saved to:   {model_path}")
print("\nRun  'mlflow ui'  in your terminal to view the dashboard.")