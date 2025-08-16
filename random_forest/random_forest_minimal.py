import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv('Data/genAI_feature_engineered.csv')

features = [col for col in df.columns if col != 'Upcoming_Event']
X = df[features]
y = df['Upcoming_Event'].astype(int)  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42,
    class_weight='balanced'
)
scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
print("Cross-validated F1 scores:", scores)
print("Mean F1 score:", scores.mean())

scores_acc = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-validated accuracy scores:", scores_acc)
print("Mean accuracy score:", scores_acc.mean())

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

importances = rf.feature_importances_
features_list = list(X.columns)
indices = importances.argsort()[::-1]

plt.figure(figsize=(12, 4))
plt.title("Feature importance")
plt.bar(range(len(features_list)), importances[indices], align="center")
plt.xticks(range(len(features_list)), [features_list[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

joblib.dump(rf, 'Models/rf_upcoming_event_model.joblib')
