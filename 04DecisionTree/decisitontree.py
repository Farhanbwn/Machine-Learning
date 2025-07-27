import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# Read the dataset into a pandas dataframe
dataset = pd.read_csv("diabetes_original.csv")

# Separate the target column
x = dataset.drop(columns='Outcome', axis=1)
y = dataset['Outcome']

# Standardize the feature data
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# Create and train the Decision Tree classifier
clf = DecisionTreeClassifier(random_state=1)
clf.fit(x_train, y_train)

# Print accuracy scores
print("Accuracy score of x_train is", clf.score(x_train, y_train))
print("Accuracy score of x_test is", clf.score(x_test, y_test))

# Additional evaluation metrics
y_pred = clf.predict(x_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    filled=True,
    feature_names=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ],
    class_names=["No Diabetes", "Diabetes"]
)
plt.title("Decision Tree Visualization")
plt.savefig("decision_tree_plot.png")  # Saves image
plt.show()  # Displays image
