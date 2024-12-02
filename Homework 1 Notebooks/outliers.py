import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the MobileNetV2 model, excluding the top layer to get feature extraction
mobilenet = tfk.applications.MobileNetV3Large(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet",
    pooling='avg',
    include_preprocessing=True,
)

# Define number of classes and output directory
num_classes = 8
output_dir = "outlier_images_per_class"

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Extract features for all images in the dataset
X = standardize(X)  # Preprocess entire dataset

# Function to find and save outliers for each class based on a distance threshold
dataset_features = mobilenet.predict(X, batch_size=32, verbose=0)
def save_outlier_images(X, y, dataset_features, distance_multiplier=2.0):
    for cls in range(num_classes):
        # Create a directory for each class to save outliers
        class_dir = os.path.join(output_dir, f"class_{cls}")
        os.makedirs(class_dir, exist_ok=True)

        # Filter features and images by class
        class_indices = np.where(y == cls)[0]
        class_features = dataset_features[class_indices]
        class_images = X[class_indices]
        
        # Calculate centroid of the class in feature space
        class_centroid = np.mean(class_features, axis=0)
        
        # Compute distances from each image in the class to the centroid
        distances = np.mean(np.square(class_features - class_centroid), axis=-1)
        
        # Calculate threshold based on mean and standard deviation of distances
        distance_threshold = np.mean(distances) + distance_multiplier * np.std(distances)
        
        # Identify outlier indices where distance exceeds the threshold
        outlier_indices = np.where(distances > distance_threshold)[0]
        
        # Save only the images that are outliers
        print(f"Class {cls} - Saving {len(outlier_indices)} outliers")
        for rank, idx in enumerate(outlier_indices):
            # Convert the image to a PIL image and save
            img = Image.fromarray((class_images[idx]).astype(np.uint8))  # Scale for display
            img_path = os.path.join(class_dir, f"outlier_{rank}.png")
            img.save(img_path)
            print(f"Saved {img_path}")

# Run the function to save outlier images for each class
save_outlier_images(X, y, dataset_features, distance_multiplier=4.0)