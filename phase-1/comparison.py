import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# Paths to saved models
XGB_MODEL_PATH = 'models/xgb_scaling_next1h.pkl'
RNN_MODEL_PATH = 'outputs/rnn_scaling_model/rnn_scaling_model.keras'

# Load XGBoost model bundle
bundle = joblib.load(XGB_MODEL_PATH)
xgb_model = bundle['model']
xgb_features = bundle['features']
xgb_threshold = float(bundle.get('chosen_threshold', 0.5))

# Load RNN model
rnn_model = load_model(RNN_MODEL_PATH)
# It is assumed RNN uses the same features as XGBoost. Change if needed!
rnn_features = xgb_features

def xgb_predict(X: pd.DataFrame):
    X = X[xgb_features]
    proba = xgb_model.predict_proba(X)[:, 1]
    label = (proba >= xgb_threshold).astype(int)
    return label, proba

def rnn_predict(X: pd.DataFrame):
    X = X[rnn_features].to_numpy()
    # reshape for RNN: (samples, timesteps=1, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    proba = rnn_model.predict(X, verbose=0).reshape(-1)
    label = (proba >= 0.5).astype(int)
    return label, proba

def evaluate_models(df: pd.DataFrame):
    y_true = df['Scaling_Next_1h']

    # Predictions from XGBoost
    xgb_labels, xgb_probas = xgb_predict(df)
    # Predictions from RNN
    rnn_labels, rnn_probas = rnn_predict(df)

    print("=== XGBoost Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_true, xgb_labels):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, xgb_probas):.4f}")
    print(classification_report(y_true, xgb_labels))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, xgb_labels))

    print("\n=== RNN Keras Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_true, rnn_labels):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, rnn_probas):.4f}")
    print(classification_report(y_true, rnn_labels))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, rnn_labels))

    print("\n=== Comparison Table ===")
    print("{:<10} {:<10} {:<10} {:<10}".format('Model', 'Accuracy', 'F1', 'ROC-AUC'))
    xgb_report = classification_report(y_true, xgb_labels, output_dict=True)
    rnn_report = classification_report(y_true, rnn_labels, output_dict=True)
    print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f}".format(
        'XGBoost', accuracy_score(y_true, xgb_labels),
        xgb_report['weighted avg']['f1-score'], roc_auc_score(y_true, xgb_probas)))
    print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f}".format(
        'RNN', accuracy_score(y_true, rnn_labels),
        rnn_report['weighted avg']['f1-score'], roc_auc_score(y_true, rnn_probas)))

if __name__ == '__main__':
    # Load your evaluation data
    # Replace with your actual CSV path
    df_eval = pd.read_csv('data/genAI_featured_trimmed.csv')
    # Validate that all features exist in df_eval
    missing_features = set(xgb_features).difference(set(df_eval.columns))
    if missing_features:
        raise ValueError(f"Missing features in evaluation data: {missing_features}")
    evaluate_models(df_eval)
