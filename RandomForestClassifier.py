import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import classification_report, confusion_matrix,precision_score,accuracy_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

df=pd.read_csv("titanic.csv")
target_column = 'Embarked'
X = df.drop(columns=[target_column])
y = df[target_column]

le = LabelEncoder()
for i in X.columns:
    if X[i].dtype == 'object':
        X[i] = le.fit_transform(X[i])
if y.dtype == 'object':
    y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(
   n_estimators=500,
    max_depth=20,
    min_samples_split=15,
    min_samples_leaf=8,
    max_features='sqrt',
    class_weight='balanced',
    bootstrap=True,
    random_state=42
)
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)


accuracy = accuracy_score(y_train,y_train_pred)
print(f"\nAccuracy of training data: {accuracy:.4f}")
accuracy1 = accuracy_score(y_test,y_test_pred)
print(f"Accuracy of testing data: {accuracy1:.4f}")

f1 = f1_score(y_train,y_train_pred, average='weighted')
print(f"\nF1 score of training data: {f1:.4f}")
f2 = f1_score(y_test,y_test_pred, average='weighted')
print(f"F1 score of testing data: {f2:.4f}")

cm = confusion_matrix(y_test, y_test_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.show()

cm_train = confusion_matrix(y_train, y_train_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm_train)
display.plot()
plt.show()
