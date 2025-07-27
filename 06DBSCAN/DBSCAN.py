import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Read the dataset
dataset = pd.read_csv("diabetes_original.csv")

# Remove the target column for unsupervised clustering
x = dataset.drop(columns='Outcome', axis=1)

# Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=1.5, min_samples=5)  # You can tune eps and min_samples
labels = dbscan.fit_predict(x_scaled)

# Add cluster labels to the dataset
dataset['Cluster'] = labels

# Plot DBSCAN clusters using two features (Glucose and BMI)
plt.figure(figsize=(10, 6))
plt.scatter(x_scaled[:, 1], x_scaled[:, 5], c=labels, cmap='plasma', s=50, alpha=0.6)
plt.xlabel('Glucose (Standardized)')
plt.ylabel('BMI (Standardized)')
plt.title('DBSCAN Clustering (Glucose vs BMI)')
plt.grid(True)
plt.savefig("dbscan_clusters_plot.png")  # Save image
plt.show()  # Show plot
