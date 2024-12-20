import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import argparse
import pickle  # Import pickle

# Parse command line arguments
parser = argparse.ArgumentParser(description='Image Clustering')
parser.add_argument('--features', type=str, help='Path to the folder containing image features in .npy files')
args = parser.parse_args()

NUM_CLUSTERS = 100

# Folder path containing the .npy feature files
feature_folder = args.features  # Use the provided feature folder path

# Number of clusters (you can adjust this)
num_clusters = NUM_CLUSTERS 

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
            
            # Flatten the feature vector if it's multidimensional
            if feature.ndim > 1:
                feature = feature.flatten()
            
            features.append(feature)
            file_names.append(file.split('.')[0])  # Remove the .npy extension

# Convert the list of feature vectors to a NumPy array
feature_matrix = np.vstack(features)

# Perform K-means clustering
print("Performing K-means clustering...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(feature_matrix)

# Save the KMeans model using pickle
model_filename = 'kmeans_model_1000.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(kmeans, file)
print(f"K-means model saved successfully to {model_filename}.")

# Create and save the DataFrame with image file names and cluster assignments
df = pd.DataFrame({'Image_File': file_names, 'Cluster': cluster_assignments})
cluster_csv_file = 'cluster_assignments_1000.csv'
df.to_csv(cluster_csv_file, index=False)
print(f"Cluster information saved to '{cluster_csv_file}'.")
