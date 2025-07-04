import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('card_transdata.csv')
target_column = 'fraud'
X = data.drop(columns=[target_column])
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight='balanced',
    bootstrap=True,
    random_state=42
)
dc = BaggingClassifier(estimator=clf, random_state=42)
dc.fit(X_train, y_train)

y_train_pred = dc.predict(X_train)
y_test_pred = dc.predict(X_test)

acc = accuracy_score(y_train, y_train_pred)
print(f"\nAccuracy of training data: {acc:.4f}")
acc1 = accuracy_score(y_test, y_test_pred)
print(f"Accuracy of testing data: {acc1:.4f}")

f1 = f1_score(y_train, y_train_pred, average='weighted')
print(f"\nF1 score of training data: {f1:.4f}")
f2 = f1_score(y_test, y_test_pred, average='weighted')
print(f"F1 score of testing data: {f2:.4f}")

cm = confusion_matrix(y_test, y_test_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.show()

cm_train = confusion_matrix(y_train, y_train_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm_train)
display.plot()
plt.show()


