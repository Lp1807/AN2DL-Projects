import numpy as np

import tensorflow as tf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
        

def horizontal_flip(x, reverse=False):
    if reverse:
        return tf.image.flip_left_right(x)
    return tf.image.flip_left_right(x)

def vertical_flip(x, reverse=False):
    if reverse:
        return tf.image.flip_up_down(x)
    return tf.image.flip_up_down(x)

def identity(x, reverse=False):
    if reverse:
        return x
    return x  

# def add_color_shift(x):
#     return tf.image.random_hue(tf.image.random_saturation(tf.image.random_brightness(x, 0.2), 0.5, 1.5), 0.2)



def predict_and_tta(model, X_test, tta_transforms=[identity, horizontal_flip, vertical_flip]):
    """
    Perform predictions with Test-Time Augmentation (TTA).

    Args:
        model (tf.keras.Model): Trained TensorFlow model.
        X_test (np.ndarray): Test dataset, assumed to be a numpy array or TensorFlow tensor.
        tta_transforms (list): List of transformation functions to apply for TTA.

    Returns:
        np.ndarray: Averaged predictions from TTA.
    """
    print("Predicting with TTA...")
    
    # Store predictions for all transformations
    tta_predictions = []

    # Loop through each TTA transformation
    for transform in tqdm(tta_transforms, desc="TTA Transforms"):
        # Apply the transformation to the test dataset
        X_transformed = tf.map_fn(lambda x: transform(x, reverse=False), X_test, dtype=X_test.dtype)

        # Predict on the transformed data
        predictions = model.predict(X_transformed)[1]

        # Reverse the predicted transformation
        predictions = tf.map_fn(lambda x: transform(x, reverse=True), predictions, dtype=predictions.dtype)

        # Ensure predictions correspond to the original orientation
        tta_predictions.append(predictions)

    # Average predictions across all transformations
    tta_predictions = np.mean(tta_predictions, axis=0)

    return tta_predictions