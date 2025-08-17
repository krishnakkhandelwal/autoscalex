import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    f1_score
)

INPUT_CSV = 'Data/genAI_featured_trimmed.csv'
OUTPUT_DIR = 'models'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'xgb_scaling_next1h.pkl')
REPORT_PATH = os.path.join(OUTPUT_DIR, 'xgb_validation_report.json')
PLOT_PR_CURVE = os.path.join(OUTPUT_DIR, 'pr_curve.png')
PLOT_CONFUSION = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
PLOT_FEATURE_IMPORTANCE = os.path.join(OUTPUT_DIR, 'feature_importance.png')

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV, parse_dates=['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

X = df.drop(columns=['Scaling_Action', 'Scaling_Next_1h', 'Datetime'])
y = df['Scaling_Next_1h'].astype(int)

split_idx = int(0.8 * len(df))
X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]

pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight = (neg / pos) if pos > 0 else 1.0

base_params = dict(
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=2,
    objective='binary:logistic',
    eval_metric='auc',
    n_jobs=-1,
    tree_method='hist',
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model = XGBClassifier(n_estimators=1000, **base_params)

supports_es = True
try:
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=50,
        early_stopping_rounds=50
    )
except TypeError:
    supports_es = False
    print("early_stopping_rounds not supported by this xgboost runtime; training without early stopping.")
    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.2,
        min_child_weight=3,
        objective='binary:logistic',
        eval_metric='auc',
        n_jobs=-1,
        tree_method='hist',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    model.fit(X_train, y_train)

proba_valid = model.predict_proba(X_valid)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_valid, proba_valid)
pr_auc = auc(recalls, precisions)

best_idx = None
best_f1 = -1
chosen_threshold = 0.5
for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
    if r >= 0.85:
        f1_tmp = 2 * p * r / (p + r + 1e-9)
        if f1_tmp > best_f1:
            best_f1 = f1_tmp
            best_idx = (p, r, t)
if best_idx is not None:
    chosen_threshold = float(best_idx[2])

pred_valid = (proba_valid >= chosen_threshold).astype(int)

report_dict = classification_report(y_valid, pred_valid, digits=3, output_dict=True)
roc_auc = roc_auc_score(y_valid, proba_valid)
acc = accuracy_score(y_valid, pred_valid)
f1 = f1_score(y_valid, pred_valid, pos_label=1)

print("Classification Report (chosen threshold):")
print(json.dumps(report_dict, indent=2))
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")
print(f"Accuracy:{acc:.4f}")
print(f"F1:      {f1:.4f}")
if supports_es:
    print("Training mode: WITH early stopping")
else:
    print("Training mode: WITHOUT early stopping (fallback)")

plt.figure(figsize=(6,4))
plt.plot(recalls, precisions, label=f'PR AUC={pr_auc:.3f}')
plt.axvline(0.85, color='red', linestyle='--', label='Recall target 0.85')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
plt.legend(); plt.tight_layout()
plt.savefig(PLOT_PR_CURVE); plt.close()

cm = confusion_matrix(y_valid, pred_valid)
plt.figure(figsize=(4,3.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NoScale','Scale'], yticklabels=['NoScale','Scale'])
plt.title('Confusion Matrix'); plt.ylabel('True'); plt.xlabel('Predicted')
plt.tight_layout(); plt.savefig(PLOT_CONFUSION); plt.close()

importance = model.get_booster().get_score(importance_type='gain')
fi = pd.Series({col: importance.get(f'f{idx}', 0.0) for idx, col in enumerate(X_train.columns)})
fi = fi.sort_values(ascending=False).head(20)

plt.figure(figsize=(8,6))
sns.barplot(x=fi.values, y=fi.index, orient='h', palette='viridis')
plt.title('Top Feature Importances (gain)')
plt.xlabel('Importance'); plt.ylabel('Feature')
plt.tight_layout(); plt.savefig(PLOT_FEATURE_IMPORTANCE); plt.close()

joblib.dump({
    'model': model,
    'chosen_threshold': chosen_threshold,
    'features': list(X_train.columns),
    'supports_early_stopping': supports_es
}, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")

summary = {
    'n_train': int(len(X_train)),
    'n_valid': int(len(X_valid)),
    'class_balance_train': {'pos': int(pos), 'neg': int(neg), 'scale_pos_weight': float(scale_pos_weight)},
    'roc_auc_valid': float(roc_auc),
    'pr_auc_valid': float(pr_auc),
    'accuracy_valid': float(acc),
    'f1_valid': float(f1),
    'chosen_threshold': float(chosen_threshold),
    'supports_early_stopping': supports_es,
    'classification_report': report_dict
}
with open(REPORT_PATH, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Saved validation report to {REPORT_PATH}")

print("Saved plots:")
print(f"- {PLOT_PR_CURVE}")
print(f"- {PLOT_CONFUSION}")
print(f"- {PLOT_FEATURE_IMPORTANCE}")
print("Section 2 complete.")
