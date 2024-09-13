import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# INPUTS. Je passe une seconde positions et d'accéléromètre.
position_vector_shape = (2, 100)
accelerometer_shape = (3, 7500)
output_shape = (1, 100)
hidden_units = 64

input_positions = tf.keras.layers.Input(shape=position_vector_shape, name="positions")
input_ax = tf.keras.layers.Input(shape=accelerometer_shape, name="accelerometer")

# mettre une première couche de convolution, et les stacker en bas? Puis passer dans un RNN?
# Pourquoi cette convolution? permet de mettre à la même taille les deux entrées.
# J'empile les deux entrées pour travailler sur les positions et l'accéléromètre en même temps.
#
conv_layer_positions = tf.keras.layers.Conv2D(input_positions)  # output
conv_layer_ax = tf.keras.layers.Conv2D(input_ax)  # output

# ces Tenseurs empilés, vont passer dans une couche de LSTM.
concatenate = tf.keras.layers.concatenate([conv_layer_positions, conv_layer_ax], axis=0)

output = tf.keras.layers.Dense(name="out")

model = tf.keras.models.Model(inputs=[input_positions, input_ax], output=output)

tf.keras.utils.plot_model(model, "decode_positions.png", show_shapes=True)
model.compile()  # Qu'est-ce que compiler un modèle?

# DUMMY DATA

dummy_positions = np.random.normal(0, 1, size=position_vector_shape)
dummy_accelerometer = np.random.normal(0, 1, size=accelerometer_shape)
dummy_output = np.random.normal(0, 1, size=output_shape)
model.fit(
            {"positions": dummy_positions, "acceleromter": accelerometer_shape},
            {"output": dummy_output},
            epochs=2,
            batch_size=32,
         )
