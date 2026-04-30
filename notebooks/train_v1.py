import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# 1. Load Data
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
file_path = os.path.join(BASE_DIR, "data", "cleaned_quikr_car.csv")
df = pd.read_csv(file_path)

# -----------------------------
# 2. Feature Engineering
# -----------------------------
current_year = 2026
df["car_age"] = current_year - df["year"]
df.drop("year", axis=1, inplace=True)

# -----------------------------
# 3. Outlier Removal (IQR method)
# -----------------------------
def remove_outliers(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[col] >= lower) & (data[col] <= upper)]

df = remove_outliers(df, "Price")
df = remove_outliers(df, "kms_driven")

# -----------------------------
# 4. Split features and target
# -----------------------------
X = df.drop("Price", axis=1)
y = df["Price"]

# -----------------------------
# 5. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# -----------------------------
# 6. Preprocessing (OneHot Encoding)
# -----------------------------
categorical_features = ["name", "company", "fuel_type"]
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

# -----------------------------
# 7. Models
# -----------------------------
lr_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

# -----------------------------
# 8. Cross Validation
# -----------------------------
lr_cv = cross_val_score(lr_model, X_train, y_train, cv=5, scoring="r2")
rf_cv = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="r2")

print("Linear Regression CV R2:", round(lr_cv.mean(), 4))
print("Random Forest CV R2:    ", round(rf_cv.mean(), 4))

# -----------------------------
# 9. Hyperparameter Tuning (Random Forest)
# -----------------------------
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth":    [None, 10, 20],
    "model__max_features": ["sqrt", "log2"],
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest RF Params:", grid_search.best_params_)
print("Best RF CV R2: ", round(grid_search.best_score_, 4))

# Use the best RF from grid search
rf_model = grid_search.best_estimator_

# -----------------------------
# 10. Train Linear Regression
# -----------------------------
lr_model.fit(X_train, y_train)

# -----------------------------
# 11. Evaluate on Test Set
# -----------------------------
def evaluate(name, y_true, y_pred):
    print(f"\n{name}")
    print("R2 Score:", round(r2_score(y_true, y_pred), 4))
    print("MAE:     ", round(mean_absolute_error(y_true, y_pred), 2))

lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

evaluate("Linear Regression", y_test, lr_pred)
evaluate("Random Forest",     y_test, rf_pred)

# -----------------------------
# 12. Save Best Model
# -----------------------------
lr_score = r2_score(y_test, lr_pred)
rf_score = r2_score(y_test, rf_pred)

if rf_score > lr_score:
    best_model = rf_model
    best_model_name = "Random Forest"
else:
    best_model = lr_model
    best_model_name = "Linear Regression"

artifact_dir = os.path.join(BASE_DIR, "artifacts")
os.makedirs(artifact_dir, exist_ok=True)

model_path = os.path.join(artifact_dir, "best_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

print(f"\nBest Model: {best_model_name}")
print(f"Saved to:   {model_path}")