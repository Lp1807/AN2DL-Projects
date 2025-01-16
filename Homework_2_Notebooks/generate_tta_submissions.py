from tta import predict_and_tta

from get_model_dualNetwork import MeanIntersectionOverUnion, ArgmaxLayer, StopGradientLayer, hybrid_loss, FocalLoss
import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import cv2
import pandas as pd
from tta import predict_and_tta

def y_to_df(y) -> pd.DataFrame:
    """Converts segmentation predictions into a DataFrame format for Kaggle."""
    n_samples = len(y)
    y_flat = y.reshape(n_samples, -1)
    df = pd.DataFrame(y_flat)
    df["id"] = np.arange(n_samples)
    cols = ["id"] + [col for col in df.columns if col != "id"]
    return df[cols]

num_classes = 5
secondLoss = FocalLoss()
mean_iou_metric = MeanIntersectionOverUnion(num_classes=num_classes, labels_to_exclude=[0])  # Exclude background class
model = tfk.models.load_model(f"model.keras",compile=False,
                                    custom_objects={'StopGradientLayer': StopGradientLayer,'ArgmaxLayer': ArgmaxLayer})
model.compile(
    loss={'output_activation_layer': hybrid_loss, 'output_layer': secondLoss},
    loss_weights= {'output_activation_layer': 1.0, 'output_layer': 1.0},
    optimizer=tf.keras.optimizers.AdamW(0.001),
    metrics={
        'output_activation_layer': [MeanIntersectionOverUnion(num_classes=5, labels_to_exclude=[0])],
        'output_layer': [MeanIntersectionOverUnion(num_classes=5, labels_to_exclude=[0])]  
    }
)

data = np.load("new_mars_clean_data.npz")
test_set = data["test_set"]
X_test = np.expand_dims(test_set, axis=-1)

# Normalize test data
X_test = X_test.astype('float32') / 255.0

# Perform predictions with TTA
predicted_labels = predict_and_tta(model, X_test)

predicted_labels = np.array(predicted_labels)
predicted_labels = np.argmax(predicted_labels, axis=-1)

submission_filename = f"submission_tta_prova.csv"
submission_df = y_to_df(predicted_labels)
submission_df.to_csv(submission_filename, index=False)


