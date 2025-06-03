import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# Create DataFrame
data = {
    "PassengerId": [1, 2, 3, 4, 5],
    "Name": ["John Smith", "Jane Doe", "Emily Johnson", "William Brown", "Anna Williams"],
    "Sex": ["male", "female", "female", "male", "female"],
    "Age": [22, 38, 26, 35, 30],
    "Pclass": [3, 1, 3, 1, 2],
    "Survived": [0, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

# Gini Index function
def gini_index(groups, classes):
    total_samples = sum([len(group) for group in groups])
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = sum(group['Survived'] == class_val) / size
            score += proportion ** 2
        gini += (1 - score) * (size / total_samples)
    return gini

# Entropy function
def entropy(group):
    size = len(group)
    if size == 0:
        return 0
    ent = 0
    for val in [0, 1]:
        p = sum(group['Survived'] == val) / size
        if p > 0:
            ent -= p * math.log2(p)
    return ent

# Information Gain function
def information_gain(df, attribute):
    total_entropy = entropy(df)
    values = df[attribute].unique()
    weighted_entropy = 0
    for value in values:
        subset = df[df[attribute] == value]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset)
    info_gain = total_entropy - weighted_entropy
    return info_gain

# Split the data by 'Sex'
male_group = df[df['Sex'] == 'male']
female_group = df[df['Sex'] == 'female']
groups = [male_group, female_group]
classes = [0, 1]

# Gini Index for split on 'Sex'
gini = gini_index(groups, classes)

# Information Gain for split on 'Sex'
info_gain = information_gain(df, 'Sex')

print("Gini Index for 'Sex':", round(gini, 4))
print("Information Gain for 'Sex':", round(info_gain, 4))


# Step 2: Encode categorical column 'Sex'
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Step 3: Define features and target
X = df[['Sex', 'Age', 'Pclass']]
y = df['Survived']

# Step 4: Create and train the decision tree
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y)

# Step 5: Visualize the decision tree as text
print("\nDecision Tree Rules:\n")
print(export_text(clf, feature_names=list(X.columns)))

# Step 6: Plot the tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.title("Decision Tree for Titanic Mini Dataset")
plt.show()"""
