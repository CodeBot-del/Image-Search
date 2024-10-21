import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Image Clustering')
parser.add_argument('--features', type=str, help='Path to the folder containing image features in .npy files')
args = parser.parse_args()

# Folder path containing the .npy feature files
feature_folder = args.features  # Use the provided feature folder path

# List to store feature vectors
features = []

# List to store file names (for reference)
file_names = []

# Load feature vectors from .npy files
print("Loading feature vectors from .npy files...")
for root, dirs, files in os.walk(feature_folder):
    for file in files:
        if file.lower().endswith('.npy'):
            file_path = os.path.join(root, file)
            feature = np.load(file_path)

            # Flatten the feature if it's not already a 1D array
            if feature.ndim > 1:
                feature = feature.flatten()

            features.append(feature)
            file_names.append(file.split('.')[0])  # Remove the .npy extension

# Convert the list of feature vectors to a NumPy array
feature_matrix = np.vstack(features)

# Reduce dimensionality using PCA
n_components = 50  # You can adjust this number based on your needs
print(f"Reducing dimensionality using PCA with {n_components} components...")
pca = PCA(n_components=n_components)
feature_matrix_reduced = pca.fit_transform(feature_matrix)

# Determine the optimal number of clusters using the Elbow Method
inertia_values = []
silhouette_scores = []
max_clusters = 20  # Adjust as needed

print("Performing K-means clustering to find the optimal number of clusters...")
for num_clusters in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(feature_matrix_reduced)  # Use the reduced feature matrix
    
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(feature_matrix_reduced, kmeans.labels_))
    print(f"Processed {num_clusters}/{max_clusters} clusters")

# Plot the Elbow Method curve
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, max_clusters + 1), inertia_values, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Within-cluster Sum of Squares)')
plt.title('Elbow Method for Optimal Number of Clusters')

plt.subplot(1, 2, 2)
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', linestyle='-', color='g')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal Number of Clusters')

plt.tight_layout()

# Save the plots as images
plt.savefig('elbow_silhouette_plots.png')

# Show the plots (optional)
plt.show()

print("Elbow Method and Silhouette Score analysis completed.")
