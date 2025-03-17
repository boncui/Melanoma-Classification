import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.data import AUTOTUNE
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping
# import kagglehub

data_train_path = 'skin-lesions/train'
data_valid_path = 'skin-lesions/valid'
data_test_path = 'skin-lesions/test'

# Checking if directories exist
for path in [data_train_path, data_valid_path, data_test_path]:
    if not os.path.exists(path):
        print(f"Directory not found: {path}")

# Defining image size and batch size
image_size = (180, 180)
batch_size = 32

# Image Augmentation
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

# Loading the training and validation datasets
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

# Preteching images to improve performance
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# The number of classes - melanoma, nevus, seborrheic keratosis
class_counts = np.zeros(3)      

for images, labels in train_ds:
    class_counts += np.sum(labels.numpy(), axis=0)          # Sum the number of images in each class

for i, count in enumerate(class_counts):
    print(f"Class {i} count: {count}")

total_images = np.sum(class_counts)
class_weights = {i: total_images / count for i, count in enumerate(class_counts)}

print(f"Class weights: {class_weights}")        # The class weights are used to balance the dataset (to counteract the class imbalance)

# Visualizing a few augmented images side by side
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 6, i + 1)  
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(np.argmax(labels[i]))
        plt.axis("off")
    
    # Appling data augmentation and displaying the augmented images
    augmented_images = data_augmentation(images)
    for i in range(9):
        ax = plt.subplot(3, 6, i + 10)  # Start second half for augmented images
        plt.imshow(augmented_images[i].numpy().astype("uint8"))
        plt.title(np.argmax(labels[i]))
        plt.axis("off")

plt.show()

# Defining the model
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    # ResNet50 base
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(180, 180, 3))
    base_model.trainable = True # Allows fine-tuning of the ResNet50 model
    x = base_model(x, training=False) # Ensure the base model is in inference mode for stable training

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x) 
    # Reduces the dimensions of the feature map to 2

    # Drops 0.3 of the neurons to prevent overfitting
    x = layers.Dropout(0.3)(x)

    # Classification layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)

# Setting input shape and number of classes
input_shape = image_size + (3,)
num_classes = 3  # melanoma, nevus, seborrheic keratosis

# Initializing and compiling the model
model = make_model(input_shape=input_shape, num_classes=num_classes)
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Training the model
epochs = 50
early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping],
    class_weight=class_weights
)

# Evaluating the model on the test dataset
test_ds = keras.utils.image_dataset_from_directory(
    data_test_path,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {test_accuracy:.2f}")

# ✅ Save the trained model
model.save("melanoma_classifier.h5")  # HDF5 format
# model.save("melanoma_classifier")  # TensorFlow format

# ✅ Verify the saved model
print("\n✅ Verifying saved model...")
loaded_model = tf.keras.models.load_model("melanoma_classifier.h5")
loaded_model.summary()
print("✅ Model successfully saved and reloaded!")

# Plotting training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


