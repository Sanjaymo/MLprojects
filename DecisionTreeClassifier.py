import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Step 1: Create the data
data = {
    'Age': [36, 42, 23, 52, 43, 44, 66, 35, 52, 35, 24, 18, 45],
    'Experience': [10, 12, 4, 4, 21, 14, 3, 14, 13, 5, 3, 3, 9],
    'Rank': [9, 4, 6, 4, 8, 5, 7, 9, 7, 9, 5, 7, 9],
    'Nationality': ['UK', 'USA', 'N', 'USA', 'USA', 'UK', 'N', 'UK', 'N', 'N', 'USA', 'UK', 'UK'],
    'Go': ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'YES', 'YES', 'YES', 'YES', 'NO', 'YES', 'YES']
}
df = pd.DataFrame(data)

# Step 2: Convert categorical columns to numeric
df['Nationality'] = df['Nationality'].map({'UK': 0, 'USA': 1, 'N': 2})
df['Go'] = df['Go'].map({'NO': 0, 'YES': 1})

# Step 3: Define features and target
X = df[['Age', 'Experience', 'Rank', 'Nationality']]  # Features
y = df['Go']                                         # Target

# Step 4: Create and train the Decision Tree
model = DecisionTreeClassifier(criterion='entropy', max_depth=3)  # Added parameters for better control
model.fit(X, y)

# Step 5: Visualize the tree with larger figure size
plt.figure(figsize=(12, 8))
tree.plot_tree(model,
               feature_names=['Age', 'Experience', 'Rank', 'Nationality'],
               class_names=['NO', 'YES'],
               filled=True,
               rounded=True,
               fontsize=10)
plt.title("Decision Tree for Candidate Selection", pad=20)
plt.show()

# Optional: Make some predictions
test_samples = pd.DataFrame({
    'Age': [30, 50],
    'Experience': [10, 15],
    'Rank': [7, 5],
    'Nationality': ['UK', 'USA']
})
test_samples['Nationality'] = test_samples['Nationality'].map({'UK': 0, 'USA': 1, 'N': 2})

predictions = model.predict(test_samples)
print("\nPredictions for test samples:")
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: {'YES' if pred == 1 else 'NO'}")
