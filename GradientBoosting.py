from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('personality_dataset.csv')
data = data.dropna()
target_column = 'Personality'
X = data.drop(columns=[target_column])
y = data[target_column]

le = LabelEncoder()
for i in X.columns:
    if X[i].dtype == 'object':
        X[i] = le.fit_transform(X[i])
if y.dtype == 'object':
    y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print("\n--- Training Data ---")
print(f"Accuracy : {accuracy_score(y_train, y_train_pred):.4f}")
print(f"F1 Score : {f1_score(y_train, y_train_pred, average='weighted'):.4f}")

print("\n--- Testing Data ---")
print(f"Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_test_pred, average='weighted'):.4f}")

imp = clf.feature_importances_
feature_names = X.columns
indices = np.argsort(imp)[::-1]

print("\n--- Feature Importances ---\n")
for i in indices:
    print(f"{feature_names[i]}: {imp[i]:.4f}")
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(imp)), imp[indices], color="blue", align="center")
plt.xticks(range(len(imp)), feature_names[indices], rotation=45, ha='right')
plt.tight_layout()
plt.show()




