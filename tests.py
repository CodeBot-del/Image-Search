import unittest
import os
import shutil
import numpy as np
import pandas as pd
from varsitysrch import VaSrch

class TestVaSrch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vasrch = VaSrch()
        cls.test_image_folder = 'test_images/'
        cls.features_folder = 'test_features/'
        cls.model_filename = 'test_kmeans_model.pkl'
        cls.csv_filename = 'test_cluster_assignments.csv'
        cls.test_image_path = os.path.join(cls.test_image_folder, 'test_image.jpg')

        # Create test folders and files
        os.makedirs(cls.test_image_folder, exist_ok=True)
        os.makedirs(cls.features_folder, exist_ok=True)

        # Add a dummy image for testing
        from PIL import Image
        img = Image.new('RGB', (224, 224), color='red')
        img.save(cls.test_image_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_image_folder)
        shutil.rmtree(cls.features_folder)
        if os.path.exists(cls.model_filename):
            os.remove(cls.model_filename)
        if os.path.exists(cls.csv_filename):
            os.remove(cls.csv_filename)

    def test_extract_features(self):
        self.vasrch.extract_features(self.test_image_folder, self.features_folder)
        features_files = os.listdir(self.features_folder)
        self.assertTrue(len(features_files) > 0)
        self.assertTrue(features_files[0].endswith('.npy'))

    def test_get_optimal_num_clusters(self):
        # Mocking optimal clusters function (no assertions here as it's visualization)
        self.vasrch.get_optimal_num_clusters(self.features_folder, max_clusters=3, n_components=2)

    def test_train_clusters(self):
        self.vasrch.train_clusters(self.features_folder, self.model_filename, self.csv_filename, num_clusters=2)
        self.assertTrue(os.path.exists(self.model_filename))
        self.assertTrue(os.path.exists(self.csv_filename))
        df = pd.read_csv(self.csv_filename)
        self.assertIn('Image_File', df.columns)
        self.assertIn('Cluster', df.columns)

    def test_search_similar_images(self):
        self.vasrch.train_clusters(self.features_folder, self.model_filename, self.csv_filename, num_clusters=2)
        similar_images = self.vasrch.search_similar_images(self.test_image_path, self.model_filename, self.csv_filename, top_n=1)
        self.assertTrue(len(similar_images) > 0)
        self.assertTrue(similar_images[0].endswith('.jpg'))

if __name__ == '__main__':
    unittest.main()
