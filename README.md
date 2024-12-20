## **Product Image Search using K-Means Clustering**

### Bare-metal implementation of the product search feature using K-Means Clustering

### Steps:
- Extract features from images using a pretrained model (VGG, ResNet, etc.)
- Find optimal number of clusters using either Elbow or Silhoutte method.
- Train a K-means clusering model on the features to group features into different clusters. Once training is complete, you will have the clustering model and a csv file containing cluster assignments information
- Use the model and the csv file to predict and retrieve similar images given a test image.

## Using the VarsitySrch Library
There are four (4) callable methods:
1. `extract_features` method which takes in the following arguments
    - `img_folder` which is the the folder containing the images you want to train on, this being your image database where you'll want to retrieve search results from.
    - `save_to` which is the folder name to where you want the features to be saved.
    We use ImageNet as our base model where we get the features at the `block5_pool` layer right before the classification layer.

2. `get_optimal_num_clusters` method which takes in the following arguments
    - `features_folder` which is the folder to where you once saved the extracted features.
    - `max_clusters` which is the maximum number of clusters you want to test on.
    - `n_components` which is the number of components to be used by the elbow and silhoutte methods.
    The method will show results from Elbow and Silhoutte method altogether on an plot. This will guide you upon choosing the number of clusters to train on. If you haven't read about Elbow and Silhoutte methods for finding optimum number of clusters in clustering algorithms please do.

3. `train_clusters` method which is the main training function and takes in the following arguments
    - `features_folder` you guessed it, the folder where we saved our extracted features.
    - `model_filename` the name you want your model to be saved in. The model will be saved as a pickel file.
    - `csv_filename` the name you want your image names and cluster assignments information (metadata) to be saved in. This will be useful when searching images later on, and even integrating to your app.
    - `num_clusters` the number of clusters you want your model to be trained on.

4. `search_similar_images`