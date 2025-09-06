# models/predict_single_row3.py
# Single-sample prediction (Row 3) with explicit scaling=0/1.

import json
import joblib
import pandas as pd

MODEL_PATH = 'models/xgb_scaling_next1h.pkl'
CUSTOM_THRESHOLD = None  # e.g., 0.60 to override saved threshold

# Row 3 values (from your screenshot, 05-06-2025 18:00).
# Make sure these keys match the exact training feature names.
row = {
    'Hour': 18,
    'Weekday': 3,
    'WeekOfMonth': 1,
    'Is_Business_Hours': 1,
    'Active_Users': 252,
    'CPU_Usage (%)': 93.86,
    'Memory_Usage (%)': 86.98,
    'Has_Event': 1,
    'Event_Payroll': 0,
    'Event_Tax': 1,
    'Event_EoM': 0,
    'Event_Risk_Score': 0.8,
    'Event_Day_Proximity_Hours': 0,
    'Saturation_Max': 93.86,
    'Active_Users_Lag1': 89,
    'CPU_Usage (%)_Lag1': 42.31,
    'Memory_Usage (%)_Lag1': 43.10,
    'Active_Users_Trend1': 163,
    'CPU_Usage (%)_Trend1': 51.55,
    'Memory_Usage (%)_Trend1': 43.88,
    'Active_Users_MA3': 198.6667,
    'CPU_Usage (%)_MA3': 77.5,
    'Memory_Usage (%)_MA3': 72.65,
}

def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle['model']
    features = bundle['features']
    saved_threshold = float(bundle.get('chosen_threshold', 0.5))
    threshold = float(CUSTOM_THRESHOLD) if CUSTOM_THRESHOLD is not None else saved_threshold

    missing = [c for c in features if c not in row]
    if missing:
        raise ValueError(f"Missing required fields for inference: {missing}")

    X = pd.DataFrame([[row[c] for c in features]], columns=features)

    proba = float(model.predict_proba(X)[:, 1][0])
    label = int(proba >= threshold)

    result = {
        'probability_scale_next_1h': proba,
        'decision_threshold': threshold,
        'predicted_label': label,
        'scaling': label  # 1 = scale, 0 = no scale
    }
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
