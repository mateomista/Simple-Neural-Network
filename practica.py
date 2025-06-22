import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

km = np.array([
    0.0,   1.0,   5.0,  10.0,  20.0,
    30.0,  40.0,  50.0,  60.0,  70.0,
    80.0,  90.0, 100.0, 120.0, 150.0,
    200.0, 250.0, 300.0, 400.0, 500.0
], dtype=np.float32)

millas = np.array([
    0.00000,   0.621371,  3.106855,  6.213710,  12.427420,
    18.641130, 24.854840, 31.068550, 37.282260, 43.495970,
    49.709680, 55.923390, 62.137100, 74.564520, 93.205650,
    124.27420, 155.34275, 186.41130, 248.54840, 310.68550
], dtype=np.float32)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
);

print("Trainning...")
history = modelo.fit(km, millas, epochs=1500, verbose=False);
print("Finish.")

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de Perdida")
plt.plot(history.history["loss"])
plt.show() 