import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from joblib import load
import pickle

# Load the pre-trained VGG model without the top classification layer
base_model = VGG16(weights='imagenet', include_top=False)

# Define a custom model that outputs the features from a chosen layer
feature_extractor = Model(inputs=base_model.input, 
                        outputs=base_model.get_layer('block5_pool').output)

# Load the trained KMeans model
with open('kmeans_model.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)

# Define a function to extract features from an image
def extract_features(image_path, feature_extractor):
    img = image.load_img(image_path, target_size=(224, 224))  # Adjust target size as needed
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = feature_extractor.predict(img_data)
    features_flattened = features.flatten()
    return features_flattened

# Function to predict the cluster for a new image
def predict_cluster(image_path, feature_extractor, kmeans_model):
    features = extract_features(image_path, feature_extractor)
    features = features.astype(np.float64)  # Convert features to double precision
    cluster = kmeans_model.predict([features])
    return cluster[0]

# Function to retrieve images from the same cluster
def retrieve_similar_images(cluster_number, csv_file='cluster_assignments.csv'):
    df = pd.read_csv(csv_file)
    similar_images = df[df['Cluster'] == cluster_number]['Image_File'].tolist()
    return similar_images

# Example usage:
image_path = 'test_image.jpg'
cluster_num = predict_cluster(image_path, feature_extractor, kmeans_model)
similar_images = retrieve_similar_images(cluster_num)
print(similar_images)
