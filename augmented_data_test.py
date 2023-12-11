import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Load CIFAR-100 data
cifar = tf.keras.datasets.cifar100
new_input_shape = (128, 128, 3)

(x_train, y_train), (x_test, y_test) = cifar.load_data()

x_train_resized = tf.image.resize(x_train, new_input_shape[:2])
x_test_resized = tf.image.resize(x_test, new_input_shape[:2])

# Define an ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# Fit the ImageDataGenerator to your training data
datagen.fit(x_train_resized)

# Number of samples to generate
num_samples_to_generate = len(x_train)

# Generate augmented samples
augmented_data = datagen.flow(x_train_resized, y_train, batch_size=32)

# Create new lists to store augmented data
augmented_x_train = []
augmented_y_train = []

# Generate and store augmented samples
generated_samples = 0
while generated_samples < num_samples_to_generate:
    new_x, new_y = augmented_data.next()
    augmented_x_train.extend(new_x)
    augmented_y_train.extend(new_y)
    generated_samples += len(new_x)

# Convert lists to numpy arrays
augmented_x_train = np.array(augmented_x_train)
augmented_y_train = np.array(augmented_y_train)

# Concatenate augmented data with original data
x_combined = np.concatenate((x_train_resized, augmented_x_train), axis=0)
y_combined = np.concatenate((y_train, augmented_y_train), axis=0)

# Define and compile the model
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=new_input_shape,
    classes=100,
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# Train the model using the combined dataset
model.fit(x_combined, y_combined, epochs=1, batch_size=32)
