import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import classification_report, confusion_matrix,precision_score,accuracy_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

df=pd.read_csv("balanced_travel_dataset.csv")
target_column = 'Preferred_Trip'
X = df.drop(columns=[target_column])
y = df[target_column]
class_counts = df["Preferred_Trip"].value_counts()

class_values = class_counts.values
majority = max(class_values)
minority = min(class_values)

le = LabelEncoder()
for i in X.columns:
    if X[i].dtype == 'object':
        X[i] = le.fit_transform(X[i])
if y.dtype == 'object':
    y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)


print("confusion matrix of training data")
cm=confusion_matrix(y_train,y_train_pred)
print(cm)

print("confusion matrix of testing data")
cm1=confusion_matrix(y_test,y_test_pred)
print(cm1)

accuracy = accuracy_score(y_train,y_train_pred)
f1 = f1_score(y_train,y_train_pred)
print(f"\nAccuracy of training data: {accuracy:.4f}")
print(  f"F1 score of training data: {f1:.4f}")

accuracy1 = accuracy_score(y_test,y_test_pred)
f11 = f1_score(y_test,y_test_pred)
print(f"\nAccuracy of testing data: {accuracy1:.4f}")
print(  f"F1 score of testing data: {f11:.4f}")
