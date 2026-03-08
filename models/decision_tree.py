# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Example data
data = {
 'Age': [25, 45, 35, 50, 23, 37, 32, 28, 40, 27],
 'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium'],
 'Student': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'],
 'Buys_Computer': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}
df = pd.DataFrame(data)

# Convert categorical features to numeric
df['Income'] = df['Income'].map({'Low': 1, 'Medium': 2, 'High': 3})
df['Student'] = df['Student'].map({'No': 0, 'Yes': 1})
df['Buys_Computer'] = df['Buys_Computer'].map({'No': 0, 'Yes': 1})

# Independent variables (features) and dependent variable (target)
X = df[['Age', 'Income', 'Student']]
y = df['Buys_Computer']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating and training the decision tree model
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
model.fit(X_train, y_train)
# Making predictions
y_pred = model.predict(X_test)
# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# Plotting the decision tree
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=['Age', 'Income', 'Student'], class_names=['No', 'Yes'], filled=True)
plt.title('Decision Tree')
plt.show()
