import tensorflow as tf

# Check the number of available GPUs
gpus = tf.config.list_physical_devices('GPU')
num_gpus = len(gpus)
print(num_gpus)

cifar = tf.keras.datasets.cifar100

# Input shape - 64x64x3
new_input_shape = (224, 224, 3)

# Load CIFAR-100 data
(x_train, y_train), (x_test, y_test) = cifar.load_data()

def resize_images(image, label):
    image = tf.image.resize(image, new_input_shape[:2])
    return image, label

train_dataset = train_dataset.map(resize_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(10000).batch(128).prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = test_dataset.map(resize_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(128).prefetch(tf.data.experimental.AUTOTUNE)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Resize images to the new input dimensions

# Create the ResNet50 model with the new input shape
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=new_input_shape,
    classes=100,  # CIFAR-100 has 100 classes
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", update_freq=1, profile_batch=(1,10),histogram_freq=1)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])


# Train the model
model.fit(x_train_resized, y_train, epochs=1, callbacks=[tensorboard_callback])
