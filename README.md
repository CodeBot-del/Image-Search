## **Product Image Search using K-Means Clustering**
This approach is suitable for e-commerce platforms where you want to return similar products that looks the same as the searched product. The approach can be used on variety of use cases and so it is implemented in such a way you can use it for whatever case you have that involves finding similar images from a collection of images given an image input.
The resulted models are lightweight and can be deployed on CPU instances with an approximately less than 500ms response time.
Training the model depends on how large your dataset is. For data collections with less or equal to 5000 images, training on CPU is okay, more than that, a GPU is recommended for faster training times.

I used ImageNet for extracting image features by the time of releasing this. The plan is to explore other base image models and add them to package where you'll have to pass them as an argument/parameter when calling the `extract_features` function.

### Bare-metal implementation of the product image search using K-Means Clustering

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

4. `search_similar_images` method which returns a cluster with similar images to the given image. The arguments are:
    - `image_path` which is the path to the image you wanna search.
    - `model_filename` which is the name of your trained model.
    - `csv_file` which is the name of the metadata csv saved during training.
    - `top_n` which is the number of image results you want to be returned.