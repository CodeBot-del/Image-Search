## **Product Image Search using K-Means Clustering**

### Bare-metal implementation of the product search feature using K-Means Clustering

### Steps:
- Extract features from images using a pretrained model (VGG, ResNet, etc.)
- Find optimal number of clusters using either Elbow or Silhoutte method.
- Train a K-means clusering model on the features to group features into different clusters. Once training is complete, you will have the clustering model and a csv file containing cluster assignments information
- Use the model and the csv file to predict and retrieve similar images given a test image.