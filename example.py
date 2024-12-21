from vasrch import extract_features, get_optimal_num_clusters, train_clusters, search_similar_images


image_path = 'test_image.jpg'
image_folder = './test_images'
features_folder = './test_features'
num_clusters = 20  #choose this number based on elbow and silhoutte plots
csv_file = "metadata.csv"
model_filename = "test_model.pkl"
top_n = 5

extract_features(image_folder, features_folder)

visualize_clusters = get_optimal_num_clusters(features_folder, max_clusters=100, n_components=10)

train_clusters(features_folder, model_filename, csv_file, num_clusters)

similar_images = search_similar_images(image_path, model_filename, csv_file, top_n)
print(f"Similar images to {image_path} are:")
for image in similar_images:
    print(image)