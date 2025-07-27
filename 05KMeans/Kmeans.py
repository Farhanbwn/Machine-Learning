import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Read the dataset
dataset = pd.read_csv("diabetes_original.csv")

# Remove the target column for unsupervised clustering
x = dataset.drop(columns='Outcome', axis=1)

# Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Apply KMeans clustering with 2 clusters (assuming 2 classes: diabetic, non-diabetic)
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(x_scaled)

# Add cluster labels to the original dataset (optional)
dataset['Cluster'] = labels

# Plot clusters using two main features for visualization (e.g., Glucose and BMI)
plt.figure(figsize=(10, 6))
plt.scatter(x_scaled[:, 1], x_scaled[:, 5], c=labels, cmap='viridis', alpha=0.6)
plt.xlabel('Glucose (Standardized)')
plt.ylabel('BMI (Standardized)')
plt.title('K-Means Clustering (2 Clusters)')
plt.grid(True)
plt.savefig("kmeans_clusters_plot.png")  # Saves the image
plt.show()  # Displays the plot
