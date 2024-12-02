import os
import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from keras.preprocessing.image import array_to_img
from imblearn.over_sampling import SMOTE
from PIL import Image
from sklearn.model_selection import train_test_split


# Load the .npz file

data = np.load('/Users/lucapagano/Developer/ANN2DL/Homework 1/Dataset/training_set_cleaned_NODUPLICATE.npz')
X = data['images']  # Shape: (num_samples, height, width, channels)
y = data['labels']  # Shape: (num_samples,)

# Converting to one-hot encoding
seed = 90
y = tfk.utils.to_categorical(y,len(np.unique(y)))

# Split data into train_val and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=seed, test_size=0.2, stratify=np.argmax(y,axis=1))


# Print shapes of the datasets
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")


# Reshape images to 2D for SMOTE
def smote_data(images, labels):
    num_samples, height, width, channels = images.shape
    X_train = images.reshape(num_samples, 96 * 96 * 3)

    X_train = X_train / 255

    # Apply SMOTE to balance the dataset
    sm = SMOTE(random_state=2)
    X_smote, y_smote = sm.fit_resample(X_train, labels)

    # Reshape SMOTE output back to image format
    X_smote_images = X_smote.reshape(-1, height, width, channels)

    X_smote_images = X_smote_images * 255
    X_smote_images = X_smote_images.astype(np.uint8)
    
    return X_smote_images, y_smote
    

X_train_smote, y_train_smote = smote_data(X_train, y_train)
X_val_smote, y_val_smote = smote_data(X_val, y_val)

print(f"X_train_smote shape: {X_train_smote.shape}, y_train shape: {y_train.shape}")
print(f"X_val_smote shape: {X_val_smote.shape}, y_val shape: {y_val.shape}")

# Compress to npz into train_images, train_labels, val_images, val_labels
np.savez('/Users/lucapagano/Desktop/Smote/smote_split_data.npz', train_images=X_train_smote, train_labels=y_train_smote, val_images=X_val_smote, val_labels=y_val_smote)
