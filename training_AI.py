# =============================================================
# INSURANCE CLAIM PREDICTION - REGRESSION MODEL
# =============================================================


# -------------------------------------------------------------
# 1. IMPORTS
# -------------------------------------------------------------

# --- Core data handling ---
import pandas as pd
import joblib

# --- Data splitting & hyperparameter search ---
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# --- Linear models (not used for now) ---
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso

# --- Tree models (not used for now) ---
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor

# --- Model currently in use ---
from xgboost import XGBRegressor


# 2. LOAD THE DATASET


data = pd.read_csv("dataset.csv")



# 3. SPLIT FEATURES (X) AND TARGET (y)


# X = every column except the one we want to predict
X = data.drop("claim", axis=1)

# y = the column we want to predict
y = data["claim"]



# 4. PREPROCESSING


# Convert text/categorical columns into 0/1 numeric columns
X = pd.get_dummies(X, dtype=int)

# Replace missing values with 0
X = X.fillna(0)



# 5. TRAIN / TEST SPLIT


# 80% for training, 20% kept aside to test on unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# 6. HYPERPARAMETER GRID


# Every combination of these values was tested by GridSearchCV
"""
param_grid = {
    "max_depth": [3, 5, 7],           # depth of each tree
    "learning_rate": [0.1, 0.3],      # how fast the model learns
    "n_estimators": [100, 500]        # number of trees
}
"""
# The result was {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 500}



# 7. LOAD THE TRAINED MODEL


# The model was already trained and saved to disk, so we just load it here
# instead of training it again.
model = joblib.load("AI_model.pkl")

# --- How the model was originally built (kept for reference) ---
"""
model = XGBRegressor(
    learning_rate=0.1,
    max_depth=7,
    n_estimators=500
)
"""



# 8. EXPORT TEST DATA FOR PREDICTION.PY

X_test.to_csv("X_test_data.csv", index=False)
y_test.to_csv("y_test_data.csv", index=False)