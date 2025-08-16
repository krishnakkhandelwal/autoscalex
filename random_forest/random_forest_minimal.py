import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("../Data/genAI_minimal.csv")

le= LabelEncoder()
df["Day"] = le.fit_transform(df["Day"])

X = df[[ "Day", "Hour", "Active_Users", "CPU_Usage (%)", "Memory_Usage (%)",
        "Business_Event_Month-End Reporting", "Business_Event_None", "Business_Event_Payroll Processing",
        "Business_Event_Tax Filing"
        ]]
y = df["Scaling_Action"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

importances = rf.feature_importances_
indices = np.argsort(importances)[::1]
features = X.columns
plt.figure()
plt.title("Feature importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

sns.pairplot(df, hue="Scaling_Action", vars=["Active_Users", "CPU_Usage (%)", "Memory_Usage (%)"])
plt.show()
