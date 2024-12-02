import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable
from sklearn.metrics import accuracy_score

data_params = {
    'batch_size': 32,
    'input_shape': (96, 96, 3),
    'num_classes': 8,
    'seed': 90
}

HYPERPARAMETERS = {
    "BATCH_SIZE": 32,
    "EPOCHS": 200,
    "LEARNING_RATE": 0.001,
    "LEARNING_DESCENT_PATIENCE": 5,
    "LEARNING_DESCENT_FACTOR": 0.5,
    "EARLY_STOPPING_PATIENCE": 10,
    "DROPOUT": 0.4,
    "LAYERS_FINE_TUNE": 250,
    "MODEL_NAME": "",
    "RAND_AUGMENT_MAGNITUDE": 0.4,
    "RAND_AUGMENT_AUGMENTATIONS_PER_IMAGE": 2
}


@register_keras_serializable(package="Custom")
class CustomSegmentationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomSegmentationLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Process inputs batch-wise using TensorFlow operations.
        """
        def extract_features_tf(sample_dd, position_weight=10):
            """
            Estrai caratteristiche dall'immagine RGB e restituisci pixel etichettati come foreground e background.
            """
            # Ottieni immagine e scribbles
            img = sample_dd['img']
            fg = sample_dd['scribble_fg']
            bg = sample_dd['scribble_bg']

            # Ottieni dimensioni immagine e appiattisci
            H, W, C = tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2]
            img_flat = tf.reshape(img, [-1, C]) / 255.0  # Normalizza valori RGB tra 0 e 1

            # Aggiungi posizioni pixel come caratteristiche aggiuntive
            positions = tf.stack(tf.meshgrid(tf.range(H), tf.range(W), indexing='ij'), axis=-1)
            positions = tf.cast(tf.reshape(positions, [-1, 2]), tf.float32)
            positions = (positions / tf.stack([tf.cast(H, tf.float32), tf.cast(W, tf.float32)])) * position_weight  # Normalizza e applica il peso
            img_flat = tf.concat([img_flat, positions], axis=1)

            # Appiattisci scribbles
            fg_flat = tf.reshape(fg, [-1])
            bg_flat = tf.reshape(bg, [-1])

            # Seleziona pixel di foreground e background
            foreground_pixels = tf.boolean_mask(img_flat, fg_flat == 1)
            background_pixels = tf.boolean_mask(img_flat, bg_flat == 1)

            # Bilancia il numero di campioni
            num_fg = tf.shape(foreground_pixels)[0]
            num_bg = tf.shape(background_pixels)[0]
            min_samples = tf.minimum(num_fg, num_bg)
            foreground_pixels = tf.random.shuffle(foreground_pixels)[:min_samples]
            background_pixels = tf.random.shuffle(background_pixels)[:min_samples]

            # Crea etichette
            labels_foreground = tf.ones([min_samples], dtype=tf.float32)
            labels_background = tf.zeros([min_samples], dtype=tf.float32)

            # Combina caratteristiche ed etichette
            features = tf.concat([foreground_pixels, background_pixels], axis=0)
            labels = tf.concat([labels_foreground, labels_background], axis=0)

            return features, labels



        def segment_image_KNN_tf(sample_dd, k=5, threshold=0.3, position_weight=10):
            """
            Segmenta immagini usando una versione KNN basata su TensorFlow.
            """
            # Ottieni dimensioni immagine
            img = sample_dd['img']
            H, W, C = tf.unstack(tf.shape(img))

            # Estrai caratteristiche ed etichette
            train_features, train_labels = extract_features_tf(sample_dd, position_weight=position_weight)

            # Normalizza caratteristiche di training
            train_features_mean = tf.reduce_mean(train_features, axis=0)
            train_features_std = tf.math.reduce_std(train_features, axis=0)
            train_features_scaled = (train_features - train_features_mean) / train_features_std

            # Prepara caratteristiche del test
            test_features = tf.reshape(img, [-1, C]) / 255.0
            positions = tf.stack(tf.meshgrid(tf.range(H), tf.range(W), indexing='ij'), axis=-1)
            positions = tf.cast(tf.reshape(positions, [-1, 2]), tf.float32)

            # Correggi la normalizzazione delle posizioni
            positions = (positions / tf.stack([tf.cast(H, tf.float32), tf.cast(W, tf.float32)])) * position_weight
            test_features = tf.concat([test_features, positions], axis=1)
            test_features_scaled = (test_features - train_features_mean) / train_features_std

            # Calcola distanze euclidee
            distances = tf.sqrt(
                tf.reduce_sum(
                    tf.square(
                        tf.expand_dims(test_features_scaled, 1) - tf.expand_dims(train_features_scaled, 0)
                    ),
                    axis=2
                )
            )

            # Trova i k vicini più vicini
            knn_indices = tf.argsort(distances, axis=1)[:, :k]
            knn_labels = tf.gather(train_labels, knn_indices)

            # Calcola probabilità media per essere foreground
            probas = tf.reduce_mean(knn_labels, axis=1)

            # Applica soglia
            labels_test = tf.cast(probas >= threshold, tf.int32)

            # Ridimensiona a dimensioni immagine
            segmented_image_mask = tf.reshape(labels_test, [H, W])

            return segmented_image_mask



        def extract_image_tf(img):
            """
            Pulisce l'immagine e seleziona pixel di foreground e background usando TensorFlow.
            """
            treshold = 0.6
            size_sample_fg = 100
            size_sample_bg = 50

            # Normalizza l'immagine
            img = tf.cast(img, tf.float32)
            img_norm = img / tf.reduce_max(img, axis=(0, 1), keepdims=True)

            # Pulisce ogni canale
            mask = img_norm < treshold
            img_cleaned = img_norm * tf.cast(mask, tf.float32)

            # Maschera per pixel non nulli
            non_zero_mask = tf.reduce_any(img_cleaned > 0, axis=-1)

            # Campionamento casuale per foreground
            fg = tf.cast(non_zero_mask, tf.uint8)
            non_zero_indices = tf.where(fg == 1)
            num_fg = tf.shape(non_zero_indices)[0]
            sampled_fg_indices = tf.random.shuffle(non_zero_indices)[:tf.minimum(size_sample_fg, num_fg)]
            selected_fg = tf.scatter_nd(
                sampled_fg_indices,
                tf.ones([tf.shape(sampled_fg_indices)[0]], dtype=tf.uint8),
                tf.cast(tf.shape(fg), tf.int64)  # Conversione a int64
            )

            # Campionamento casuale per background
            bg = 1 - fg
            bg_non_zero_indices = tf.where(bg == 1)
            num_bg = tf.shape(bg_non_zero_indices)[0]
            sampled_bg_indices = tf.random.shuffle(bg_non_zero_indices)[:tf.minimum(size_sample_bg, num_bg)]
            selected_bg = tf.scatter_nd(
                sampled_bg_indices,
                tf.ones([tf.shape(sampled_bg_indices)[0]], dtype=tf.uint8),
                tf.cast(tf.shape(bg), tf.int64)  # Conversione a int64
            )
            # Prepara il dizionario per la segmentazione
            image_test = {
                'img': img,
                'scribble_fg': selected_fg,
                'scribble_bg': selected_bg
            }

            # Genera la maschera segmentata
            segmented_image_mask = segment_image_KNN_tf(image_test, k=5, position_weight=100)

            # Applica la maschera segmentata all'immagine originale
            segmented_pixels = tf.where(
                tf.expand_dims(segmented_image_mask, axis=-1) == 1,
                img,
                tf.zeros_like(img)
            )

            return segmented_pixels


        def preprocess_image_for_extract_tf(image):
            """
            Preprocessa un'immagine ridimensionandola, convertendola in RGB e normalizzandola.
            """
            # Resize to 96x96
            image_resized = tf.image.resize(image, [96, 96])

            # Ensure the image is in RGB format
            if image_resized.shape[-1] == 1:  # If grayscale, convert to RGB
                image_rgb = tf.image.grayscale_to_rgb(image_resized)
            else:
                image_rgb = image_resized

            # Normalize pixel values to [0, 1]
            image_normalized = tf.clip_by_value(image_rgb / 255.0, 0.0, 1.0)

            # Remove batch dimension if present
            if len(image_normalized.shape) == 4 and image_normalized.shape[0] == 1:
                image_normalized = tf.squeeze(image_normalized, axis=0)

            return image_normalized

        def process_image(image):
            """
            Process a single image: preprocessing and segmentation.
            """
            # Preprocess the image (e.g., resizing)
            preprocessed_image = preprocess_image_for_extract_tf(image)
            # Apply segmentation logic (using TensorFlow-native functions)
            segmented_image = extract_image_tf(preprocessed_image)
            return segmented_image

        # Use tf.map_fn to apply process_image to each item in the batch
        outputs = tf.map_fn(process_image, inputs, dtype=tf.float32)
        return outputs


class Model:
    def __init__(self):
        input_shape = (96, 96, 3)
        self.neural_network = tfk.models.load_model('SegmentationprovahistoryBackup.keras', custom_objects={
            "CustomSegmentationLayer": CustomSegmentationLayer
        })

    def build_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        MODEL_IMPORTED = tfk.applications.EfficientNetV2S(
            include_top=False,
            weights="imagenet",
            input_shape=data_params['input_shape'],
            pooling="avg",
            classes=8,
            classifier_activation="softmax",
            include_preprocessing=True,
            name="efficientnetv2-s",
        )

        for layer in MODEL_IMPORTED.layers:
            layer.trainable = False

        x = CustomSegmentationLayer()(inputs)
        x = MODEL_IMPORTED(x)

        x = tfkl.Dense(128, activation='relu')(x)
        x = tfkl.Dropout(HYPERPARAMETERS['DROPOUT'])(x)
        x = tfkl.Dense(64, activation='relu')(x)
        x = tfkl.Dropout(HYPERPARAMETERS['DROPOUT'])(x)
        outputs = tfkl.Dense(data_params['num_classes'], activation='softmax')(x)

        model = tfk.Model(inputs=inputs, outputs=outputs)
        print(model.summary())

        return model

    def predict(self, X):
        preds = self.neural_network.predict(X)
        if len(preds.shape) == 2:
            preds = np.argmax(preds, axis=1)
        return preds

    def save_weights(self, filepath):
        self.neural_network.save_weights(filepath)
        print(f"Weights saved to {filepath}")


# model = Model()
# # Carica il dataset
# data = np.load("Dataset/training_set_cleaned_NODUPLICATE.npz")
# X = data['images']  # Supponendo che i dati siano memorizzati in una chiave chiamata 'X'
# y_true = data['labels']  # Supponendo che le etichette siano memorizzate in una chiave chiamata 'y'

# # Inizializza il model
# # Effettua le predizioni
# y_pred = model.predict(X)

# # Calcola l'accuratezza
# accuracy = accuracy_score(y_true, y_pred)
# print(f"Accuratezza: {accuracy * 100:.2f}%")