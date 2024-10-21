import os
import numpy as np
import argparse
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Parse command line arguments
parser = argparse.ArgumentParser(description='Image Feature Extraction')
parser.add_argument('--img_folder', type=str, help='Path to the folder containing images')
parser.add_argument('--save_to', type=str, help='Path to the folder where features will be saved')
args = parser.parse_args()

# Load the pre-trained VGG model without the top classification layer
base_model = VGG16(weights='imagenet', include_top=False)

# Define a custom model that outputs the features from a chosen layer
feature_extractor = Model(inputs=base_model.input, 
                        outputs=base_model.get_layer('block5_pool').output)

# Define a function to extract features from an image
def extract_features(image_path, feature_extractor):
    img = image.load_img(image_path, target_size=(224, 224))  # Adjust target size as needed
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = feature_extractor.predict(img_data)
    return features

# Folder path containing the images
folder_path = args.img_folder  # Use the provided image folder path

# Define a folder to save the feature files
output_folder = args.save_to  # Use the provided features folder path

# Loop through all image files in the folder and its subfolders
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            # Get the full path of the image file
            img_path = os.path.join(root, file)
            
            # Extract the file name (without extension) to use as the feature file name
            file_name = os.path.splitext(file)[0]
            
            # Extract features, save as .npy file
            img_features = extract_features(img_path, feature_extractor)
            np.save(output_folder + file_name + '.npy', img_features)

print("Feature extraction and saving completed.")
