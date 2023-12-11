import tensorflow as tf
import ssl

ssl._create_default_https_context = ssl._create_unverified_context



cifar = tf.keras.datasets.cifar100

new_input_shape = (128,128,3)

(x_train, y_train), (x_test, y_test) = cifar.load_data()

x_train_resized = tf.image.resize(x_train, new_input_shape[:2])
x_test_resized = tf.image.resize(x_test, new_input_shape[:2])

model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=new_input_shape,
    classes=100,)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

model.fit(x_train_resized, y_train, epochs=1, batch_size=32)
