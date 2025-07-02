import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

actual = np.random.binomial(1, 0.9, size=1000)
pred = np.random.binomial(1, 0.9, size=1000)
cm = confusion_matrix(actual, pred)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
cm_display.plot()
plt.show()

accuracy = accuracy_score(actual, pred)
print("Accuracy:", accuracy)
presion = precision_score(actual, pred)
print("presion:",presion)
recall = recall_score(actual, pred)
print("recall:",recall)
f1 = f1_score(actual, pred)
print("f1:",f1)

cm = confusion_matrix(actual, pred)
tn,tp,fn,fp = cm.ravel()
print(f"tn : {tn},tp : {tp},fn : {fn},fp : {fp}")
misclass = (fn+fp)/(tn+tp+fn+fp)
print(f"missclassification rate : {misclass}")
fpr = fp/(tn+fp)
print(f"false positive rate : {fpr}")
prevalance = (fn+tp)/(tp+tn+fp+fn)
print(f"Prevalance : {prevalance}")

