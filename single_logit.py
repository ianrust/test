import tensorflow as tf


my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation='relu'),
	    tf.keras.layers.Dense(2, activation='softmax')
    ]
)

model.build((1, None, None, 3))
print(model.summary())
model.save('./model.h5')