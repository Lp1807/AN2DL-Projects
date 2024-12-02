# Do model ensemble

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

model1_91 = tfk.models.load_model('/content/drive/MyDrive/0,91.keras', compile=False)
model1_91._name = 'model1_92'
model2_92 = tfk.models.load_model('/content/drive/MyDrive/0,92.keras', compile=False)
model2_92._name = 'model2_92'
# Ensembling them
models = [model1_91, model2_92]
model_input = tf.keras.Input(shape=(96, 96, 3))
model_outputs = [model(model_input) for model in models]

ensemble_output = tfkl.average(model_outputs)
ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)

ensemble_model.compile(optimizer=tfk.optimizers.AdamW(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

tfk.utils.plot_model(ensemble_model, show_shapes=True, show_layer_names=True, to_file='ensemble_model.png')
ensemble_model.save('/content/drive/MyDrive/ensemble_91_92.keras')