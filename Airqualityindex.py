# ============================================================
# AQI FORECASTING USING MACHINE LEARNING
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# LOAD DATA
# ============================================================

print("="*60)
print("AIR QUALITY INDEX FORECASTING SYSTEM")
print("="*60)

# Sample synthetic dataset

np.random.seed(42)

N = 3000

df = pd.DataFrame({
    "PM25": np.random.normal(40,15,N),
    "PM10": np.random.normal(70,20,N),
    "NO2": np.random.normal(35,10,N),
    "CO": np.random.normal(1.5,0.5,N),
    "SO2": np.random.normal(12,4,N),
    "TEMP": np.random.normal(25,5,N),
    "HUMIDITY": np.random.normal(60,15,N)
})

df["AQI"] = (
    0.45*df["PM25"]
    +0.20*df["PM10"]
    +0.15*df["NO2"]
    +np.random.normal(0,5,N)
)

print("\nDataset Shape :",df.shape)

# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop("AQI",axis=1)
y = df["AQI"]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

# ============================================================
# MODELS
# ============================================================

models = {
    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=150,
            random_state=42
        )
}

results = {}

print("\nTraining Models...\n")

for name,model in models.items():

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test,pred)

    rmse = np.sqrt(
        mean_squared_error(y_test,pred)
    )

    r2 = r2_score(y_test,pred)

    cv = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    ).mean()

    results[name] = {
        "model":model,
        "MAE":mae,
        "RMSE":rmse,
        "R2":r2,
        "CV":cv,
        "pred":pred
    }

    print(name)
    print("MAE :",round(mae,2))
    print("RMSE:",round(rmse,2))
    print("R2  :",round(r2,4))
    print("CV  :",round(cv,4))
    print("-"*40)

# ============================================================
# BEST MODEL
# ============================================================

best_name = max(
    results,
    key=lambda x: results[x]["R2"]
)

best_model = results[best_name]["model"]

print("\nBest Model :",best_name)

# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8,5))

plt.scatter(
    y_test,
    results[best_name]["pred"],
    alpha=0.5
)

plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")

plt.show()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

if hasattr(best_model,"feature_importances_"):

    importance = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    )

    importance.sort_values().plot(
        kind="barh",
        figsize=(8,5)
    )

    plt.title("Feature Importance")
    plt.show()

# ============================================================
# SAMPLE PREDICTION
# ============================================================

sample = X_test[0].reshape(1,-1)

prediction = best_model.predict(sample)

print("\nPredicted AQI :",round(prediction[0],2))

print("\nProject Completed Successfully")
