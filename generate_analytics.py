# ============================================================
# ACCIDENT HOTSPOT PREDICTION SYSTEM
# ANALYTICS GENERATOR
# ============================================================
#
# Input:
#     accident_prediction_india.csv
#
# Creates:
#     graphs/
#     metrics/
#     results/
#     reports/
#     deployment/
#     static/graphs/
#
# Run:
#     python generate_analytics.py
# ============================================================

import os
import json
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "accident_prediction_india.csv"
)

GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DEPLOYMENT_DIR = os.path.join(BASE_DIR, "deployment")
STATIC_GRAPHS_DIR = os.path.join(
    BASE_DIR,
    "static",
    "graphs"
)


# ============================================================
# 2. CREATE FOLDERS
# ============================================================

folders = [
    GRAPHS_DIR,
    METRICS_DIR,
    RESULTS_DIR,
    REPORTS_DIR,
    DEPLOYMENT_DIR,
    STATIC_GRAPHS_DIR,
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("\n============================================================")
print("ACCIDENT HOTSPOT PREDICTION - ANALYTICS GENERATOR")
print("============================================================\n")


# ============================================================
# 3. CHECK DATASET
# ============================================================

if not os.path.exists(DATA_FILE):
    print("ERROR: Dataset not found!")
    print()
    print("Expected file:")
    print(DATA_FILE)
    print()
    print("Put accident_prediction_india.csv in the same folder")
    print("as generate_analytics.py.")
    raise SystemExit(1)


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Dataset loaded successfully.")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}\n")


# ============================================================
# 5. BASIC CLEANING
# ============================================================

df = df.copy()

# Remove completely empty columns
df = df.dropna(axis=1, how="all")

# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

print(f"Rows after cleaning: {len(df)}")


# ============================================================
# 6. FIND TARGET COLUMN
# ============================================================

target_candidates = [
    "Accident Severity",
    "accident severity",
    "Accident_Severity",
    "Severity",
    "severity",
]

TARGET = None

for column in target_candidates:
    if column in df.columns:
        TARGET = column
        break

if TARGET is None:
    print("\nERROR: Could not find accident severity column.")
    print("Available columns:")
    for column in df.columns:
        print(" -", column)
    raise SystemExit(1)

print(f"Target column: {TARGET}\n")


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def save_plot(filename):
    """
    Save graph into graphs/ and static/graphs/.
    """
    graph_path = os.path.join(GRAPHS_DIR, filename)
    static_path = os.path.join(STATIC_GRAPHS_DIR, filename)

    plt.tight_layout()
    plt.savefig(
        graph_path,
        dpi=150,
        bbox_inches="tight"
    )

    shutil.copy2(
        graph_path,
        static_path
    )

    plt.close()

    print("Created:", filename)


def find_column(possible_names):
    """
    Find a column using case-insensitive matching.
    """
    lower_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in possible_names:
        key = name.strip().lower()

        if key in lower_map:
            return lower_map[key]

    return None


def safe_numeric(column):
    """
    Convert a column to numeric where possible.
    """
    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 8. DETECT IMPORTANT COLUMNS
# ============================================================

weather_col = find_column([
    "Weather Conditions",
    "Weather Condition",
    "Weather",
])

road_col = find_column([
    "Road Conditions",
    "Road Condition",
    "Road",
])

vehicle_col = find_column([
    "Vehicle Type",
    "Vehicle Types",
    "Vehicle",
])

lighting_col = find_column([
    "Lighting Conditions",
    "Lighting Condition",
    "Lighting",
])

alcohol_col = find_column([
    "Alcohol Involvement",
    "Alcohol",
    "Alcohol Involved",
])

age_col = find_column([
    "Driver Age",
    "Age",
])

speed_col = find_column([
    "Speed Limit",
    "Speed",
])

date_col = find_column([
    "Date",
    "Accident Date",
])

time_col = find_column([
    "Time",
    "Accident Time",
])

year_col = find_column([
    "Year",
    "Accident Year",
])

month_col = find_column([
    "Month",
    "Accident Month",
])


# ============================================================
# 9. SEVERITY DISTRIBUTION
# ============================================================

print("\nGenerating graphs...\n")

severity_counts = (
    df[TARGET]
    .astype(str)
    .value_counts()
)

plt.figure(figsize=(8, 5))

severity_counts.plot(
    kind="bar"
)

plt.title("Accident Severity Distribution")
plt.xlabel("Accident Severity")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=0)

save_plot("01_severity_distribution.png")


# ============================================================
# 10. WEATHER CONDITIONS
# ============================================================

if weather_col:

    counts = (
        df[weather_col]
        .astype(str)
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(9, 5))

    counts.plot(kind="bar")

    plt.title("Accidents by Weather Conditions")
    plt.xlabel("Weather Condition")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=35, ha="right")

    save_plot("02_weather_conditions.png")


# ============================================================
# 11. ROAD CONDITIONS
# ============================================================

if road_col:

    counts = (
        df[road_col]
        .astype(str)
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(9, 5))

    counts.plot(kind="bar")

    plt.title("Accidents by Road Conditions")
    plt.xlabel("Road Condition")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=35, ha="right")

    save_plot("03_road_conditions.png")


# ============================================================
# 12. VEHICLE TYPES
# ============================================================

if vehicle_col:

    counts = (
        df[vehicle_col]
        .astype(str)
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(9, 5))

    counts.plot(kind="bar")

    plt.title("Accidents by Vehicle Type")
    plt.xlabel("Vehicle Type")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=35, ha="right")

    save_plot("04_vehicle_types.png")


# ============================================================
# 13. LIGHTING CONDITIONS
# ============================================================

if lighting_col:

    counts = (
        df[lighting_col]
        .astype(str)
        .value_counts()
    )

    plt.figure(figsize=(8, 5))

    counts.plot(kind="bar")

    plt.title("Accidents by Lighting Conditions")
    plt.xlabel("Lighting Condition")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=30, ha="right")

    save_plot("05_lighting_conditions.png")


# ============================================================
# 14. ALCOHOL INVOLVEMENT
# ============================================================

if alcohol_col:

    counts = (
        df[alcohol_col]
        .astype(str)
        .value_counts()
    )

    plt.figure(figsize=(7, 5))

    counts.plot(kind="bar")

    plt.title("Accidents by Alcohol Involvement")
    plt.xlabel("Alcohol Involvement")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=0)

    save_plot("06_alcohol_involvement.png")


# ============================================================
# 15. DATE / MONTH ANALYSIS
# ============================================================

if date_col:

    dates = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    month_counts = (
        dates
        .dt.month
        .value_counts()
        .sort_index()
    )

    if len(month_counts) > 0:

        plt.figure(figsize=(9, 5))

        month_counts.plot(
            kind="bar"
        )

        plt.title("Accidents by Month")
        plt.xlabel("Month")
        plt.ylabel("Number of Accidents")

        save_plot("07_accidents_by_month.png")


# ============================================================
# 16. YEAR ANALYSIS
# ============================================================

if year_col:

    years = pd.to_numeric(
        df[year_col],
        errors="coerce"
    )

elif date_col:

    years = pd.to_datetime(
        df[date_col],
        errors="coerce"
    ).dt.year

else:
    years = None


if years is not None:

    year_data = pd.DataFrame({
        "Year": years,
        "Severity": df[TARGET].astype(str)
    }).dropna()

    if len(year_data) > 0:

        year_severity = pd.crosstab(
            year_data["Year"],
            year_data["Severity"]
        )

        plt.figure(figsize=(10, 5))

        year_severity.plot(
            kind="bar",
            ax=plt.gca()
        )

        plt.title("Accident Severity by Year")
        plt.xlabel("Year")
        plt.ylabel("Number of Accidents")
        plt.xticks(rotation=45)

        save_plot("08_severity_by_year.png")


# ============================================================
# 17. SEVERITY BY WEATHER
# ============================================================

if weather_col:

    table = pd.crosstab(
        df[weather_col].astype(str),
        df[TARGET].astype(str)
    )

    table = table.head(10)

    plt.figure(figsize=(10, 6))

    table.plot(
        kind="bar",
        ax=plt.gca()
    )

    plt.title("Accident Severity by Weather")
    plt.xlabel("Weather Condition")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=35, ha="right")

    save_plot("09_severity_by_weather.png")


# ============================================================
# 18. SEVERITY BY ROAD CONDITION
# ============================================================

if road_col:

    table = pd.crosstab(
        df[road_col].astype(str),
        df[TARGET].astype(str)
    )

    table = table.head(10)

    plt.figure(figsize=(10, 6))

    table.plot(
        kind="bar",
        ax=plt.gca()
    )

    plt.title("Accident Severity by Road Condition")
    plt.xlabel("Road Condition")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=35, ha="right")

    save_plot("10_severity_by_road_condition.png")


# ============================================================
# 19. SEVERITY BY LIGHTING
# ============================================================

if lighting_col:

    table = pd.crosstab(
        df[lighting_col].astype(str),
        df[TARGET].astype(str)
    )

    plt.figure(figsize=(9, 5))

    table.plot(
        kind="bar",
        ax=plt.gca()
    )

    plt.title("Accident Severity by Lighting")
    plt.xlabel("Lighting Condition")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=30, ha="right")

    save_plot("11_severity_by_lighting.png")


# ============================================================
# 20. SEVERITY BY ALCOHOL
# ============================================================

if alcohol_col:

    table = pd.crosstab(
        df[alcohol_col].astype(str),
        df[TARGET].astype(str)
    )

    plt.figure(figsize=(8, 5))

    table.plot(
        kind="bar",
        ax=plt.gca()
    )

    plt.title("Accident Severity by Alcohol Involvement")
    plt.xlabel("Alcohol Involvement")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=0)

    save_plot("12_severity_by_alcohol.png")


# ============================================================
# 21. DRIVER AGE DISTRIBUTION
# ============================================================

if age_col:

    age_values = safe_numeric(age_col).dropna()

    if len(age_values) > 0:

        plt.figure(figsize=(9, 5))

        plt.hist(
            age_values,
            bins=15
        )

        plt.title("Driver Age Distribution")
        plt.xlabel("Driver Age")
        plt.ylabel("Number of Drivers")

        save_plot("13_driver_age_distribution.png")


# ============================================================
# 22. SPEED LIMIT DISTRIBUTION
# ============================================================

if speed_col:

    speed_values = safe_numeric(speed_col).dropna()

    if len(speed_values) > 0:

        plt.figure(figsize=(9, 5))

        speed_values.value_counts().sort_index().plot(
            kind="bar"
        )

        plt.title("Speed Limit Distribution")
        plt.xlabel("Speed Limit")
        plt.ylabel("Number of Accidents")
        plt.xticks(rotation=45)

        save_plot("14_speed_limit_distribution.png")


# ============================================================
# 23. NUMERIC CORRELATION HEATMAP
# ============================================================

numeric_df = df.select_dtypes(
    include=np.number
)

if numeric_df.shape[1] >= 2:

    correlation = numeric_df.corr()

    plt.figure(figsize=(11, 8))

    plt.imshow(
        correlation,
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title("Numeric Feature Correlation")

    save_plot("15_correlation_heatmap.png")


# ============================================================
# 24. MACHINE LEARNING DATA PREPARATION
# ============================================================

print("\nPreparing machine learning analysis...")

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(str)

# Remove obvious ID columns
id_columns = []

for column in X.columns:

    name = str(column).lower()

    if (
        name == "id"
        or name.endswith("_id")
        or name.startswith("id_")
        or "record id" in name
        or "accident id" in name
    ):
        id_columns.append(column)

if id_columns:
    X = X.drop(
        columns=id_columns,
        errors="ignore"
    )


# ============================================================
# 25. HANDLE TARGET CLASSES
# ============================================================

valid_rows = y.notna()

X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)

class_names = sorted(
    y.unique().tolist()
)

print("\nSeverity classes:")

for class_name in class_names:
    print(
        f" - {class_name}: "
        f"{int((y == class_name).sum())}"
    )


# ============================================================
# 26. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 27. PREPROCESSING
# ============================================================

numeric_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=np.number
).columns.tolist()


numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 28. RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest...")

rf_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
        )
    ]
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = rf_model.predict(
    X_test
)


# ============================================================
# 29. LOGISTIC REGRESSION MODEL
# ============================================================

print("Training Logistic Regression...")

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "scaler",
            StandardScaler(
                with_mean=False
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_predictions = logistic_model.predict(
    X_test
)


# ============================================================
# 30. MODEL METRICS FUNCTION
# ============================================================

def calculate_metrics(
    model_name,
    actual,
    predicted
):

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(
            actual,
            predicted
        ),
        "Precision_Macro": precision_score(
            actual,
            predicted,
            average="macro",
            zero_division=0
        ),
        "Recall_Macro": recall_score(
            actual,
            predicted,
            average="macro",
            zero_division=0
        ),
        "F1_Macro": f1_score(
            actual,
            predicted,
            average="macro",
            zero_division=0
        )
    }


rf_metrics = calculate_metrics(
    "Random Forest",
    y_test,
    rf_predictions
)

logistic_metrics = calculate_metrics(
    "Logistic Regression",
    y_test,
    logistic_predictions
)


# ============================================================
# 31. SAVE MODEL COMPARISON
# ============================================================

model_metrics = pd.DataFrame([
    rf_metrics,
    logistic_metrics
])

model_metrics.to_csv(
    os.path.join(
        METRICS_DIR,
        "model_comparison.csv"
    ),
    index=False
)

print("\nModel comparison:")
print(model_metrics.to_string(index=False))


# ============================================================
# 32. SAVE INDIVIDUAL MODEL METRICS
# ============================================================

pd.DataFrame([rf_metrics]).to_csv(
    os.path.join(
        METRICS_DIR,
        "random_forest_metrics.csv"
    ),
    index=False
)

pd.DataFrame([logistic_metrics]).to_csv(
    os.path.join(
        METRICS_DIR,
        "logistic_regression_metrics.csv"
    ),
    index=False
)


# ============================================================
# 33. MODEL COMPARISON GRAPH
# ============================================================

metric_names = [
    "Accuracy",
    "Precision_Macro",
    "Recall_Macro",
    "F1_Macro"
]

x = np.arange(len(metric_names))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    [
        rf_metrics[m]
        for m in metric_names
    ],
    width,
    label="Random Forest"
)

plt.bar(
    x + width / 2,
    [
        logistic_metrics[m]
        for m in metric_names
    ],
    width,
    label="Logistic Regression"
)

plt.title("Machine Learning Model Comparison")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.xticks(
    x,
    metric_names,
    rotation=20
)

plt.legend()

save_plot("16_model_comparison.png")


# ============================================================
# 34. CONFUSION MATRIX - RANDOM FOREST
# ============================================================

rf_cm = confusion_matrix(
    y_test,
    rf_predictions,
    labels=class_names
)

plt.figure(figsize=(7, 6))

plt.imshow(
    rf_cm,
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=30
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j,
            i,
            rf_cm[i, j],
            ha="center",
            va="center"
        )

save_plot(
    "17_random_forest_confusion_matrix.png"
)


# ============================================================
# 35. CONFUSION MATRIX - LOGISTIC REGRESSION
# ============================================================

logistic_cm = confusion_matrix(
    y_test,
    logistic_predictions,
    labels=class_names
)

plt.figure(figsize=(7, 6))

plt.imshow(
    logistic_cm,
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=30
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j,
            i,
            logistic_cm[i, j],
            ha="center",
            va="center"
        )

save_plot(
    "17_logistic_regression_confusion_matrix.png"
)


# ============================================================
# 36. FEATURE IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")

try:

    fitted_preprocessor = (
        rf_model
        .named_steps["preprocessor"]
    )

    rf_classifier = (
        rf_model
        .named_steps["classifier"]
    )

    feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
    )

    importances = (
        rf_classifier
        .feature_importances_
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

    top_features = importance_df.head(15)

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )

    plt.title(
        "Top Accident Prediction Features"
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    save_plot(
        "18_feature_importance.png"
    )

except Exception as error:

    print(
        "Feature importance could not be generated:",
        error
    )


# ============================================================
# 37. SAVE PREDICTION RESULTS
# ============================================================

prediction_results = X_test.copy()

prediction_results["Actual Severity"] = (
    y_test.values
)

prediction_results["Random Forest Prediction"] = (
    rf_predictions
)

prediction_results["Logistic Regression Prediction"] = (
    logistic_predictions
)

prediction_results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "model_predictions.csv"
    ),
    index=False
)


# ============================================================
# 38. CLASSIFICATION REPORTS
# ============================================================

rf_report = classification_report(
    y_test,
    rf_predictions,
    output_dict=True,
    zero_division=0
)

logistic_report = classification_report(
    y_test,
    logistic_predictions,
    output_dict=True,
    zero_division=0
)

rf_report_df = pd.DataFrame(
    rf_report
).transpose()

logistic_report_df = pd.DataFrame(
    logistic_report
).transpose()

rf_report_df.to_csv(
    os.path.join(
        REPORTS_DIR,
        "random_forest_classification_report.csv"
    )
)

logistic_report_df.to_csv(
    os.path.join(
        REPORTS_DIR,
        "logistic_regression_classification_report.csv"
    )
)


# ============================================================
# 39. COMBINED MODEL REPORT
# ============================================================

report_rows = []

for class_name in class_names:

    rf_row = rf_report.get(
        class_name,
        {}
    )

    lr_row = logistic_report.get(
        class_name,
        {}
    )

    report_rows.append({
        "Severity": class_name,

        "RF_Precision": rf_row.get(
            "precision",
            0
        ),

        "RF_Recall": rf_row.get(
            "recall",
            0
        ),

        "RF_F1": rf_row.get(
            "f1-score",
            0
        ),

        "LR_Precision": lr_row.get(
            "precision",
            0
        ),

        "LR_Recall": lr_row.get(
            "recall",
            0
        ),

        "LR_F1": lr_row.get(
            "f1-score",
            0
        )
    })


combined_report = pd.DataFrame(
    report_rows
)

combined_report.to_csv(
    os.path.join(
        REPORTS_DIR,
        "model_report.csv"
    ),
    index=False
)


# ============================================================
# 40. ANALYTICS SUMMARY
# ============================================================

summary = {
    "dataset": {
        "file": "accident_prediction_india.csv",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target": TARGET
    },

    "severity_distribution": {
        str(key): int(value)
        for key, value
        in severity_counts.items()
    },

    "classes": class_names,

    "models": {
        "Random Forest": {
            key: float(value)
            for key, value in rf_metrics.items()
            if key != "Model"
        },

        "Logistic Regression": {
            key: float(value)
            for key, value in logistic_metrics.items()
            if key != "Model"
        }
    },

    "columns_detected": {
        "weather": weather_col,
        "road_condition": road_col,
        "vehicle_type": vehicle_col,
        "lighting": lighting_col,
        "alcohol": alcohol_col,
        "driver_age": age_col,
        "speed_limit": speed_col,
        "date": date_col,
        "time": time_col,
        "year": year_col,
        "month": month_col
    },

    "generated_graphs": [
        filename
        for filename in os.listdir(GRAPHS_DIR)
        if filename.lower().endswith(".png")
    ]
}


with open(
    os.path.join(
        RESULTS_DIR,
        "analytics_summary.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# 41. SAMPLE PREDICTION INPUT
# ============================================================

sample_input = {}

for column in X.columns:

    if pd.api.types.is_numeric_dtype(
        X[column]
    ):

        value = X[column].median()

        if pd.isna(value):
            value = 0

        sample_input[column] = float(value)

    else:

        values = (
            X[column]
            .dropna()
            .astype(str)
        )

        if len(values) > 0:
            sample_input[column] = values.iloc[0]
        else:
            sample_input[column] = ""


with open(
    os.path.join(
        DEPLOYMENT_DIR,
        "sample_prediction_input.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        sample_input,
        file,
        indent=4
    )


# ============================================================
# 42. DATASET SUMMARY CSV
# ============================================================

dataset_summary = pd.DataFrame({
    "Column": df.columns,
    "Data Type": [
        str(df[column].dtype)
        for column in df.columns
    ],
    "Missing Values": [
        int(df[column].isna().sum())
        for column in df.columns
    ],
    "Unique Values": [
        int(df[column].nunique())
        for column in df.columns
    ]
})

dataset_summary.to_csv(
    os.path.join(
        METRICS_DIR,
        "dataset_summary.csv"
    ),
    index=False
)


# ============================================================
# 43. SEVERITY SUMMARY CSV
# ============================================================

severity_summary = pd.DataFrame({
    "Severity": severity_counts.index,
    "Accident Count": severity_counts.values
})

severity_summary["Percentage"] = (
    severity_summary["Accident Count"]
    / severity_summary["Accident Count"].sum()
    * 100
)

severity_summary.to_csv(
    os.path.join(
        METRICS_DIR,
        "severity_summary.csv"
    ),
    index=False
)


# ============================================================
# 44. GENERATE README
# ============================================================

readme_content = """
ACCIDENT HOTSPOT PREDICTION - ANALYTICS
=======================================

This folder contains analytics generated from:

accident_prediction_india.csv

Generated folders
-----------------

graphs/
    Accident analytics graphs and charts.

metrics/
    Dataset and machine-learning metrics.

results/
    Prediction results, feature importance and analytics summary.

reports/
    Classification and model reports.

deployment/
    Sample prediction input.

static/graphs/
    Copies of generated graphs for Flask dashboard use.


Important
---------

This analytics generator does NOT modify:

- app.py
- index.html
- dashboard.js
- CSS files
- existing ML model files
- existing prediction output

The generated analytics are separate and can be connected
to the existing dashboard later.
"""

with open(
    os.path.join(
        REPORTS_DIR,
        "README.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        readme_content.strip()
    )


# ============================================================
# 45. FINAL OUTPUT
# ============================================================

print("\n============================================================")
print("ANALYTICS GENERATION COMPLETE")
print("============================================================")

print("\nCreated folders:")

for folder in folders:
    print(" -", os.path.relpath(
        folder,
        BASE_DIR
    ))

print("\nGenerated graph files:")

graph_files = sorted(
    [
        f
        for f in os.listdir(GRAPHS_DIR)
        if f.lower().endswith(".png")
    ]
)

for filename in graph_files:
    print(" -", filename)

print("\nGenerated analytics files:")

for directory in [
    METRICS_DIR,
    RESULTS_DIR,
    REPORTS_DIR,
    DEPLOYMENT_DIR,
]:

    for filename in sorted(
        os.listdir(directory)
    ):

        print(
            " -",
            os.path.relpath(
                os.path.join(
                    directory,
                    filename
                ),
                BASE_DIR
            )
        )

print("\n============================================================")
print("Your existing Flask prediction system was NOT modified.")
print("============================================================\n")

print("Next command:")
print("python app.py")
print()