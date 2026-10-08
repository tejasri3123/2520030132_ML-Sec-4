import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_FILE = "accident_prediction_india.csv"

print("=" * 60)
print("ACCIDENT SEVERITY PREDICTION - MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nDataset columns:")
for i, column in enumerate(df.columns, 1):
    print(f"{i}. {column}")


# ============================================================
# 2. TARGET
# ============================================================

TARGET = "Accident Severity"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found!")

X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 3. PROCESS TIME
# ============================================================

def parse_time(value):
    try:
        value = str(value).strip()
        parts = value.split(":")

        hour = int(parts[0])
        minute = int(parts[1])

        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return np.nan, np.nan

        total_minutes = hour * 60 + minute

        return total_minutes, hour

    except Exception:
        return np.nan, np.nan


time_data = X["Time of Day"].apply(parse_time)

X["Time_Minutes"] = time_data.apply(lambda x: x[0])
X["Hour"] = time_data.apply(lambda x: x[1])

X["Weekend"] = (
    X["Day of Week"]
    .astype(str)
    .str.strip()
    .isin(["Saturday", "Sunday"])
    .astype(int)
)


# ============================================================
# 4. FEATURE DEFINITIONS
# ============================================================

numeric_features = [
    "Year",
    "Number of Vehicles Involved",
    "Number of Casualties",
    "Number of Fatalities",
    "Speed Limit (km/h)",
    "Driver Age",
    "Time_Minutes",
    "Hour",
    "Weekend"
]

categorical_features = [
    "State Name",
    "City Name",
    "Month",
    "Day of Week",
    "Time of Day",
    "Vehicle Type Involved",
    "Weather Conditions",
    "Road Type",
    "Road Condition",
    "Lighting Conditions",
    "Traffic Control Presence",
    "Driver Gender",
    "Driver License Status",
    "Alcohol Involvement",
    "Accident Location Details"
]


# ============================================================
# 5. CHECK FEATURES
# ============================================================

all_features = numeric_features + categorical_features

missing_features = [
    feature for feature in all_features
    if feature not in X.columns
]

if missing_features:
    print("\nMissing features:")
    print(missing_features)
    raise ValueError("Some required features are missing from dataset.")


# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print("\nRandom Forest Accuracy:")
print(f"{rf_accuracy * 100:.2f}%")

print("\nRandom Forest Classification Report:")
print(
    classification_report(
        y_test,
        rf_predictions
    )
)

print("Random Forest Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        rf_predictions
    )
)


# ============================================================
# 9. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

lr_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    ))
])

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_accuracy = accuracy_score(
    y_test,
    lr_predictions
)

print("\nLogistic Regression Accuracy:")
print(f"{lr_accuracy * 100:.2f}%")

print("\nLogistic Regression Classification Report:")
print(
    classification_report(
        y_test,
        lr_predictions
    )
)

print("Logistic Regression Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        lr_predictions
    )
)


# ============================================================
# 10. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs("models", exist_ok=True)


# ============================================================
# 11. SAVE MODELS
# ============================================================

joblib.dump(
    rf_model,
    "models/random_forest_model.pkl"
)

joblib.dump(
    lr_model,
    "models/logistic_regression_model.pkl"
)

print("\nModels saved successfully!")


# ============================================================
# 12. VERIFY CLASSES
# ============================================================

print("\nRandom Forest classes:")
print(rf_model.classes_)

print("\nLogistic Regression classes:")
print(lr_model.classes_)


# ============================================================
# 13. CREATE WEBSITE METADATA
# ============================================================

metadata = {
    "target": TARGET,

    "classes": [
        "Minor",
        "Serious",
        "Fatal"
    ],

    "numeric_features": numeric_features,

    "categorical_features": categorical_features,

    "all_input_features": [
        "State Name",
        "City Name",
        "Year",
        "Month",
        "Day of Week",
        "Time of Day",
        "Number of Vehicles Involved",
        "Vehicle Type Involved",
        "Number of Casualties",
        "Number of Fatalities",
        "Weather Conditions",
        "Road Type",
        "Road Condition",
        "Lighting Conditions",
        "Traffic Control Presence",
        "Speed Limit (km/h)",
        "Driver Age",
        "Driver Gender",
        "Driver License Status",
        "Alcohol Involvement",
        "Accident Location Details"
    ],

    "options": {}
}


# ============================================================
# 14. GET ACTUAL DROPDOWN VALUES FROM DATASET
# ============================================================

for column in categorical_features:

    if column in df.columns:

        values = (
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        metadata["options"][column] = sorted(values)


# Keep Month in calendar order
metadata["options"]["Month"] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]


# Keep Day of Week in correct order
metadata["options"]["Day of Week"] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


# ============================================================
# 15. SAVE METADATA
# ============================================================

with open(
    "feature_metadata.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")

print(
    f"Random Forest Accuracy: "
    f"{rf_accuracy * 100:.2f}%"
)

print(
    f"Logistic Regression Accuracy: "
    f"{lr_accuracy * 100:.2f}%"
)

print("\nClasses:")
print(rf_model.classes_)

print("\nCreated files:")
print("✓ models/random_forest_model.pkl")
print("✓ models/logistic_regression_model.pkl")
print("✓ feature_metadata.json")

print("\nNext step: Flask backend + modern dashboard")