from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('personality_dataset.csv')
data = data.dropna()
target_column = 'Personality'
X = data.drop(columns=[target_column, 'Drained_after_socializing'])
y = data[target_column]

le = LabelEncoder()
for i in X.columns:
    if X[i].dtype == 'object':
        X[i] = le.fit_transform(X[i])
    if y.dtype == 'object':
        y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)
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

cm = confusion_matrix(y_test, y_test_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.title("Testing")
plt.show()

cm_train = confusion_matrix(y_train, y_train_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm_train)
display.plot()
plt.title("Training")
plt.show()




