from flask import Flask, render_template, request, jsonify
import joblib
import json
import pandas as pd
import sys
import os
from datetime import datetime

# Configure UTF-8 encoding for standard output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RF_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_model.pkl"
)

LR_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "logistic_regression_model.pkl"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "feature_metadata.json"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "accident_prediction_india.csv"
)

PREDICTIONS_PATH = os.path.join(
    BASE_DIR,
    "results",
    "model_predictions.csv"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR,
    "results",
    "feature_importance.csv"
)


# ============================================================
# LOAD ANALYTICS DATA (Distribution, Actual vs Pred, Features, Weather)
# ============================================================

def load_analytics_data():
    distribution = {
        "labels": ["0% - 20%", "20% - 40%", "40% - 60%", "60% - 80%", "80% - 100%"],
        "counts": [175, 754, 1142, 768, 161]
    }

    actual_vs_pred = {
        "labels": [f"Point {i+1}" for i in range(50)],
        "actual": [],
        "predicted": []
    }

    feature_importance = {
        "labels": ["Speed Limit", "Driver Age", "Casualties", "Fatalities", "Year", "Vehicles Involved", "Traffic Signs", "Alcohol"],
        "importance": [4.22, 3.96, 3.32, 2.86, 2.79, 2.54, 1.22, 1.19]
    }

    weather_severity = {
        "conditions": ["Clear", "Rainy", "Foggy", "Stormy", "Hazy"],
        "minor": [202, 224, 174, 231, 203],
        "serious": [182, 210, 207, 177, 205],
        "fatal": [190, 197, 195, 203, 200]
    }

    try:
        if os.path.exists(DATASET_PATH):
            df_raw = pd.read_csv(DATASET_PATH)
            casc = df_raw["Number of Casualties"].fillna(0) / 10.0
            fatc = df_raw["Number of Fatalities"].fillna(0) / 5.0
            spd = (df_raw["Speed Limit (km/h)"].fillna(50) - 30).clip(lower=0) / 90.0
            veh = (df_raw["Number of Vehicles Involved"].fillna(1) - 1).clip(lower=0) / 4.0
            risk_load = (casc * 0.35 + fatc * 0.40 + spd * 0.15 + veh * 0.10) * 100
            cuts = pd.cut(
                risk_load,
                bins=[0, 20, 40, 60, 80, 100],
                labels=distribution["labels"],
                include_lowest=True
            )
            distribution["counts"] = cuts.value_counts().reindex(distribution["labels"], fill_value=0).tolist()

            # Weather crosstab
            ct = pd.crosstab(df_raw["Weather Conditions"], df_raw["Accident Severity"])
            top_w = ["Clear", "Rainy", "Foggy", "Stormy", "Hazy"]
            weather_severity["conditions"] = top_w
            weather_severity["minor"] = [int(ct.loc[w, "Minor"]) if w in ct.index and "Minor" in ct.columns else 0 for w in top_w]
            weather_severity["serious"] = [int(ct.loc[w, "Serious"]) if w in ct.index and "Serious" in ct.columns else 0 for w in top_w]
            weather_severity["fatal"] = [int(ct.loc[w, "Fatal"]) if w in ct.index and "Fatal" in ct.columns else 0 for w in top_w]

            print("✓ Analytics distribution & weather computed from dataset")
    except Exception as e:
        print("⚠ Analytics distribution computation error:", e)

    try:
        if os.path.exists(PREDICTIONS_PATH):
            df_pred = pd.read_csv(PREDICTIONS_PATH).head(50)
            base_map = {"Minor": 650, "Serious": 1550, "Fatal": 2350}
            for i, row in df_pred.iterrows():
                casc_val = float(row.get("Number of Casualties", 0))
                fatc_val = float(row.get("Number of Fatalities", 0))
                spd_val = float(row.get("Speed Limit (km/h)", 50))
                veh_val = float(row.get("Number of Vehicles Involved", 1))

                act_base = base_map.get(str(row.get("Actual Severity", "Minor")), 1000)
                pred_base = base_map.get(str(row.get("Random Forest Prediction", "Minor")), 1000)

                act_score = act_base + casc_val * 40 + fatc_val * 100 + (spd_val / 120.0) * 180 + (veh_val - 1) * 30
                pred_score = pred_base + casc_val * 40 + fatc_val * 100 + (spd_val / 120.0) * 180 + (veh_val - 1) * 30

                actual_vs_pred["actual"].append(round(act_score, 1))
                actual_vs_pred["predicted"].append(round(pred_score, 1))

            print(f"✓ Analytics 50 actual vs predicted points loaded")
    except Exception as e:
        print("⚠ Analytics actual vs predicted computation error:", e)

    try:
        if os.path.exists(FEATURE_IMPORTANCE_PATH):
            df_feat = pd.read_csv(FEATURE_IMPORTANCE_PATH).head(8)
            label_cleanup = {
                "numeric__Speed Limit (km/h)": "Speed Limit",
                "numeric__Driver Age": "Driver Age",
                "numeric__Number of Casualties": "Casualties",
                "numeric__Number of Fatalities": "Fatalities",
                "numeric__Year": "Year",
                "numeric__Number of Vehicles Involved": "Vehicles Involved",
                "categorical__Traffic Control Presence_Signs": "Traffic Signs",
                "categorical__Alcohol Involvement_Yes": "Alcohol Presence"
            }
            feature_importance["labels"] = [
                label_cleanup.get(f, f.replace("numeric__", "").replace("categorical__", ""))
                for f in df_feat["Feature"]
            ]
            feature_importance["importance"] = [
                round(float(v) * 100, 2) for v in df_feat["Importance"]
            ]
            print("✓ Analytics feature importance loaded")
    except Exception as e:
        print("⚠ Analytics feature importance error:", e)

    return {
        "distribution": distribution,
        "actual_vs_pred": actual_vs_pred,
        "feature_importance": feature_importance,
        "weather_severity": weather_severity
    }

ANALYTICS_DATA = load_analytics_data()


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading accident severity models...")

try:
    rf_model = joblib.load(RF_MODEL_PATH)
    print("✓ Random Forest model loaded")
except Exception as e:
    rf_model = None
    print("✗ Random Forest model error:", e)


try:
    lr_model = joblib.load(LR_MODEL_PATH)
    print("✓ Logistic Regression model loaded")
except Exception as e:
    lr_model = None
    print("✗ Logistic Regression model error:", e)


# ============================================================
# LOAD FEATURE METADATA
# ============================================================

try:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        feature_metadata = json.load(f)

    print("✓ Feature metadata loaded")

except Exception as e:
    feature_metadata = {}
    print("⚠ Feature metadata could not be loaded:", e)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_int(value, default=0):
    """Safely convert a value to integer."""

    try:
        if value is None or str(value).strip() == "":
            return default

        return int(float(value))

    except (ValueError, TypeError):
        return default


def safe_float(value, default=0.0):
    """Safely convert a value to float."""

    try:
        if value is None or str(value).strip() == "":
            return default

        return float(value)

    except (ValueError, TypeError):
        return default


def convert_time_to_minutes(time_value):
    """
    Convert HH:MM time into minutes from midnight.
    """

    if not time_value:
        return 0

    try:
        parts = str(time_value).split(":")

        hour = int(parts[0])
        minute = int(parts[1])

        return hour * 60 + minute

    except (ValueError, IndexError):
        return 0


def get_hour(time_value):
    """
    Extract hour from HH:MM.
    """

    if not time_value:
        return 0

    try:
        return int(str(time_value).split(":")[0])

    except (ValueError, IndexError):
        return 0


def get_weekend(day):
    """
    Return 1 for Saturday/Sunday, otherwise 0.
    """

    if not day:
        return 0

    return 1 if str(day).lower() in ["saturday", "sunday"] else 0


def get_severity_class(prediction):
    """
    Convert prediction into dashboard severity class.
    """

    prediction = str(prediction).strip()

    if prediction.lower() == "fatal":
        return "Fatal"

    if prediction.lower() == "serious":
        return "Serious"

    return "Minor"


# ============================================================
# RISK INDICATORS
# ============================================================

def calculate_risk_indicators(form_data):

    indicators = []

    vehicles = safe_int(
        form_data.get("Number of Vehicles Involved")
    )

    casualties = safe_int(
        form_data.get("Number of Casualties")
    )

    fatalities = safe_int(
        form_data.get("Number of Fatalities")
    )

    speed_limit = safe_float(
        form_data.get("Speed Limit (km/h)")
    )

    driver_age = safe_int(
        form_data.get("Driver Age")
    )

    alcohol = str(
        form_data.get("Alcohol Involvement", "")
    ).strip().lower()

    license_status = str(
        form_data.get("Driver License Status", "")
    ).strip().lower()

    lighting = str(
        form_data.get("Lighting Conditions", "")
    ).strip().lower()

    road_condition = str(
        form_data.get("Road Condition", "")
    ).strip().lower()

    weather = str(
        form_data.get("Weather Conditions", "")
    ).strip().lower()


    # --------------------------------------------------------
    # Fatality indicator
    # --------------------------------------------------------

    if fatalities > 0:

        indicators.append({
            "name": "Fatality involvement",
            "level": "Critical",
            "description":
                f"{fatalities} fatalit{'y' if fatalities == 1 else 'ies'} "
                "were reported in the accident."
        })

    else:

        indicators.append({
            "name": "Fatality involvement",
            "level": "Low",
            "description":
                "No fatalities were reported for this accident."
        })


    # --------------------------------------------------------
    # Casualty indicator
    # --------------------------------------------------------

    if casualties >= 10:

        indicators.append({
            "name": "High casualty count",
            "level": "High",
            "description":
                f"{casualties} casualties were reported."
        })

    elif casualties >= 4:

        indicators.append({
            "name": "Elevated casualty count",
            "level": "Medium",
            "description":
                f"{casualties} casualties were reported."
        })

    else:

        indicators.append({
            "name": "Casualty count",
            "level": "Low",
            "description":
                f"{casualties} casualties were reported."
        })


    # --------------------------------------------------------
    # Vehicle involvement
    # --------------------------------------------------------

    if vehicles >= 5:

        indicators.append({
            "name": "Multiple vehicle involvement",
            "level": "High",
            "description":
                f"{vehicles} vehicles were involved."
        })

    elif vehicles >= 3:

        indicators.append({
            "name": "Multiple vehicle involvement",
            "level": "Medium",
            "description":
                f"{vehicles} vehicles were involved."
        })

    else:

        indicators.append({
            "name": "Vehicle involvement",
            "level": "Low",
            "description":
                f"{vehicles} vehicle{'s' if vehicles != 1 else ''} "
                "were involved."
        })


    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    if speed_limit >= 100:

        indicators.append({
            "name": "High speed environment",
            "level": "High",
            "description":
                f"Speed limit is {speed_limit:.0f} km/h."
        })

    elif speed_limit >= 70:

        indicators.append({
            "name": "Moderate speed environment",
            "level": "Medium",
            "description":
                f"Speed limit is {speed_limit:.0f} km/h."
        })

    else:

        indicators.append({
            "name": "Speed environment",
            "level": "Low",
            "description":
                f"Speed limit is {speed_limit:.0f} km/h."
        })


    # --------------------------------------------------------
    # Alcohol
    # --------------------------------------------------------

    if alcohol == "yes":

        indicators.append({
            "name": "Alcohol involvement",
            "level": "Critical",
            "description":
                "Alcohol involvement was reported."
        })

    else:

        indicators.append({
            "name": "Alcohol involvement",
            "level": "Low",
            "description":
                "No alcohol involvement was reported."
        })


    # --------------------------------------------------------
    # Driver license
    # --------------------------------------------------------

    if license_status in ["expired", "suspended", "none"]:

        indicators.append({
            "name": "License compliance",
            "level": "High",
            "description":
                "The recorded license status indicates "
                "a compliance concern."
        })

    else:

        indicators.append({
            "name": "License compliance",
            "level": "Low",
            "description":
                "The recorded driver license status is valid."
        })


    # --------------------------------------------------------
    # Lighting
    # --------------------------------------------------------

    if lighting == "dark":

        indicators.append({
            "name": "Low-light conditions",
            "level": "High",
            "description":
                "The accident occurred under dark lighting conditions."
        })

    elif lighting in ["dawn/dusk", "dawn", "dusk"]:

        indicators.append({
            "name": "Reduced visibility",
            "level": "Medium",
            "description":
                "The accident occurred around dawn or dusk."
        })


    # --------------------------------------------------------
    # Road condition
    # --------------------------------------------------------

    if road_condition in [
        "wet",
        "damaged",
        "under construction",
        "snow/ice"
    ]:

        indicators.append({
            "name": "Road condition concern",
            "level": "Medium",
            "description":
                f"Recorded road condition: "
                f"{form_data.get('Road Condition', 'Unknown')}."
        })


    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    if weather in [
        "rainy",
        "foggy",
        "stormy",
        "snowy"
    ]:

        indicators.append({
            "name": "Adverse weather",
            "level": "Medium",
            "description":
                f"Recorded weather: "
                f"{form_data.get('Weather Conditions', 'Unknown')}."
        })


    return indicators


# ============================================================
# ACCIDENT PROFILE
# ============================================================

def calculate_profile(form_data):

    return {
        "location": {
            "state": form_data.get(
                "State Name",
                "Unknown"
            ),

            "city": form_data.get(
                "City Name",
                "Unknown"
            )
        },

        "vehicles": safe_int(
            form_data.get(
                "Number of Vehicles Involved"
            )
        ),

        "casualties": safe_int(
            form_data.get(
                "Number of Casualties"
            )
        ),

        "fatalities": safe_int(
            form_data.get(
                "Number of Fatalities"
            )
        ),

        "weather": form_data.get(
            "Weather Conditions",
            "Unknown"
        ),

        "road_condition": form_data.get(
            "Road Condition",
            "Unknown"
        ),

        "lighting": form_data.get(
            "Lighting Conditions",
            "Unknown"
        )
    }


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_model_input(form_data):

    time_value = form_data.get(
        "Time of Day",
        ""
    )

    day_value = form_data.get(
        "Day of Week",
        ""
    )


    data = {

        "State Name":
            form_data.get("State Name", ""),

        "City Name":
            form_data.get("City Name", ""),

        "Year":
            safe_int(form_data.get("Year")),

        "Month":
            form_data.get("Month", ""),

        "Day of Week":
            day_value,

        "Time of Day":
            time_value,

        "Number of Vehicles Involved":
            safe_int(
                form_data.get(
                    "Number of Vehicles Involved"
                )
            ),

        "Vehicle Type Involved":
            form_data.get(
                "Vehicle Type Involved",
                ""
            ),

        "Number of Casualties":
            safe_int(
                form_data.get(
                    "Number of Casualties"
                )
            ),

        "Number of Fatalities":
            safe_int(
                form_data.get(
                    "Number of Fatalities"
                )
            ),

        "Weather Conditions":
            form_data.get(
                "Weather Conditions",
                ""
            ),

        "Road Type":
            form_data.get(
                "Road Type",
                ""
            ),

        "Road Condition":
            form_data.get(
                "Road Condition",
                ""
            ),

        "Lighting Conditions":
            form_data.get(
                "Lighting Conditions",
                ""
            ),

        "Traffic Control Presence":
            form_data.get(
                "Traffic Control Presence",
                ""
            ),

        "Speed Limit (km/h)":
            safe_float(
                form_data.get(
                    "Speed Limit (km/h)"
                )
            ),

        "Driver Age":
            safe_int(
                form_data.get(
                    "Driver Age"
                )
            ),

        "Driver Gender":
            form_data.get(
                "Driver Gender",
                ""
            ),

        "Driver License Status":
            form_data.get(
                "Driver License Status",
                ""
            ),

        "Alcohol Involvement":
            form_data.get(
                "Alcohol Involvement",
                ""
            ),

        "Accident Location Details":
            form_data.get(
                "Accident Location Details",
                ""
            ),

        # Derived features
        "Time_Minutes":
            convert_time_to_minutes(
                time_value
            ),

        "Hour":
            get_hour(
                time_value
            ),

        "Weekend":
            get_weekend(
                day_value
            )
    }


    return pd.DataFrame([data])


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        analytics=ANALYTICS_DATA
    )


# ============================================================
# ANALYTICS API
# ============================================================

@app.route("/api/analytics", methods=["GET"])
def get_analytics():

    return jsonify({
        "status": "success",
        "data": ANALYTICS_DATA
    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Collect submitted form data
        # ----------------------------------------------------

        form_data = request.form.to_dict()


        # ----------------------------------------------------
        # Check models
        # ----------------------------------------------------

        if rf_model is None:
            raise RuntimeError(
                "Random Forest model is not loaded."
            )

        if lr_model is None:
            raise RuntimeError(
                "Logistic Regression model is not loaded."
            )


        # ----------------------------------------------------
        # Create model input
        # ----------------------------------------------------

        model_input = create_model_input(
            form_data
        )


        # ----------------------------------------------------
        # Random Forest prediction
        # ----------------------------------------------------

        rf_raw_prediction = rf_model.predict(
            model_input
        )[0]

        rf_prediction = get_severity_class(
            rf_raw_prediction
        )


        # ----------------------------------------------------
        # Logistic Regression prediction
        # ----------------------------------------------------

        lr_raw_prediction = lr_model.predict(
            model_input
        )[0]

        lr_prediction = get_severity_class(
            lr_raw_prediction
        )


        # ----------------------------------------------------
        # Fatality safety rule
        #
        # If fatalities are entered and the Random Forest
        # predicts Minor, display Serious instead.
        #
        # This prevents a known fatal accident from being
        # displayed as Minor in the dashboard.
        # ----------------------------------------------------

        fatalities = safe_int(
            form_data.get(
                "Number of Fatalities"
            )
        )

        if fatalities > 0 and rf_prediction == "Minor":

            rf_prediction = "Serious"


        # ----------------------------------------------------
        # Primary dashboard prediction
        # ----------------------------------------------------

        prediction = rf_prediction


        # ----------------------------------------------------
        # Random Forest probabilities
        # ----------------------------------------------------

        rf_probabilities_raw = rf_model.predict_proba(
            model_input
        )[0]


        rf_classes = [
            get_severity_class(c)
            for c in rf_model.classes_
        ]


        probabilities = {
            "Minor": 0.0,
            "Serious": 0.0,
            "Fatal": 0.0
        }


        for class_name, probability in zip(
            rf_classes,
            rf_probabilities_raw
        ):

            if class_name in probabilities:

                probabilities[class_name] = (
                    float(probability) * 100
                )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = max(
            probabilities.values()
        )


        # ----------------------------------------------------
        # Model agreement
        # ----------------------------------------------------

        if rf_prediction == lr_prediction:

            model_agreement = "AGREE"

        else:

            model_agreement = "DIFFER"


        # ----------------------------------------------------
        # Risk indicators
        # ----------------------------------------------------

        risk_indicators = calculate_risk_indicators(
            form_data
        )


        # ----------------------------------------------------
        # Accident profile
        #
        # IMPORTANT:
        # The HTML expects the variable name "profile".
        # ----------------------------------------------------

        profile = calculate_profile(
            form_data
        )


        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp = datetime.now().strftime(
            "%d %b %Y, %H:%M"
        )


        # ----------------------------------------------------
        # Input summary
        # ----------------------------------------------------

        input_summary = {}

        input_summary["State Name"] = form_data.get(
            "State Name",
            ""
        )

        input_summary["City Name"] = form_data.get(
            "City Name",
            ""
        )

        input_summary["Year"] = form_data.get(
            "Year",
            ""
        )

        input_summary["Month"] = form_data.get(
            "Month",
            ""
        )

        input_summary["Day of Week"] = form_data.get(
            "Day of Week",
            ""
        )

        input_summary["Time of Day"] = form_data.get(
            "Time of Day",
            ""
        )

        input_summary["Number of Vehicles Involved"] = form_data.get(
            "Number of Vehicles Involved",
            ""
        )

        input_summary["Vehicle Type Involved"] = form_data.get(
            "Vehicle Type Involved",
            ""
        )

        input_summary["Number of Casualties"] = form_data.get(
            "Number of Casualties",
            ""
        )

        input_summary["Number of Fatalities"] = form_data.get(
            "Number of Fatalities",
            ""
        )

        input_summary["Weather Conditions"] = form_data.get(
            "Weather Conditions",
            ""
        )

        input_summary["Road Type"] = form_data.get(
            "Road Type",
            ""
        )

        input_summary["Road Condition"] = form_data.get(
            "Road Condition",
            ""
        )

        input_summary["Lighting Conditions"] = form_data.get(
            "Lighting Conditions",
            ""
        )

        input_summary["Traffic Control Presence"] = form_data.get(
            "Traffic Control Presence",
            ""
        )

        input_summary["Speed Limit (km/h)"] = form_data.get(
            "Speed Limit (km/h)",
            ""
        )

        input_summary["Driver Age"] = form_data.get(
            "Driver Age",
            ""
        )

        input_summary["Driver Gender"] = form_data.get(
            "Driver Gender",
            ""
        )

        input_summary["Driver License Status"] = form_data.get(
            "Driver License Status",
            ""
        )

        input_summary["Alcohol Involvement"] = form_data.get(
            "Alcohol Involvement",
            ""
        )

        input_summary["Accident Location Details"] = form_data.get(
            "Accident Location Details",
            ""
        )


        # ----------------------------------------------------
        # Render dashboard
        # ----------------------------------------------------

        return render_template(
            "index.html",

            prediction=prediction,

            confidence=confidence,

            probabilities=probabilities,

            rf_prediction=rf_prediction,

            lr_prediction=lr_prediction,

            model_agreement=model_agreement,

            risk_indicators=risk_indicators,

            profile=profile,

            timestamp=timestamp,

            input_summary=input_summary,

            analytics=ANALYTICS_DATA
        )


    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )

        return (
            f"<h2>Prediction error:</h2>"
            f"<pre>{str(e)}</pre>",
            500
        )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return (
        "<h2>404 - Page not found</h2>",
        404
    )


@app.errorhandler(500)
def internal_server_error(error):

    return (
        "<h2>500 - Internal server error</h2>",
        500
    )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("ACCIDENT SEVERITY PREDICTION SYSTEM")
    print("Classes: Minor / Serious / Fatal")
    print("Random Forest + Logistic Regression")
    print("Server: http://127.0.0.1:5000")
    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )