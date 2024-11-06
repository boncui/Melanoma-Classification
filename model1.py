import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.data import AUTOTUNE
import kagglehub

# Define paths for train, validation, and test sets
data_train_path = 'skin-lesions/train'
data_valid_path = 'skin-lesions/valid'
data_test_path = 'skin-lesions/test'

# Check if directories exist
for path in [data_train_path, data_valid_path, data_test_path]:
    if not os.path.exists(path):
        print(f"Directory not found: {path}")

# Define image size and batch size
image_size = (180, 180)
batch_size = 32

# Data Augmentation layers
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomWidth(0.2),
        layers.RandomHeight(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ]
)

# Load the datasets without validation split
train_ds = keras.utils.image_dataset_from_directory(
    data_train_path,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'  # Multiclass labels
)

val_ds = keras.utils.image_dataset_from_directory(
    data_valid_path,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# Apply prefetching for performance
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Visualize a few augmented images side by side
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    # Display the original images
    for i in range(9):
        ax = plt.subplot(3, 6, i + 1)  # 3 rows and 6 columns (for original and augmented)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(np.argmax(labels[i]))
        plt.axis("off")
    
    # Apply data augmentation and display the augmented images
    augmented_images = data_augmentation(images)
    for i in range(9):
        ax = plt.subplot(3, 6, i + 10)  # Start second half for augmented images
        plt.imshow(augmented_images[i].numpy().astype("uint8"))
        plt.title(np.argmax(labels[i]))
        plt.axis("off")

plt.show()

# Define the model
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Data augmentation block
    x = data_augmentation(inputs)
    # Rescale pixel values to [0, 1]
    x = layers.Rescaling(1.0 / 255)(x)
    
    # Convolutional base
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)  # Dropout to prevent overfitting
    
    # Classification layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

# Set input shape and number of classes
input_shape = image_size + (3,)
num_classes = 3  # melanoma, nevus, seborrheic keratosis

# Initialize and compile the model
model = make_model(input_shape=input_shape, num_classes=num_classes)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display the model structure
model.summary()

# Train the model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
)

# Evaluate the model on the test dataset
test_ds = keras.utils.image_dataset_from_directory(
    data_test_path,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {test_accuracy:.2f}")

# Plot training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
