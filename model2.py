import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.mixed_precision import set_global_policy
import keras_cv

# Enable mixed precision for faster training
set_global_policy('mixed_float16')

# Paths to datasets
data_train_path = 'skin-lesions/train'
data_valid_path = 'skin-lesions/valid'
data_test_path = 'skin-lesions/test'

# Checking if directories exist
for path in [data_train_path, data_valid_path, data_test_path]:
    if not os.path.exists(path):
        print(f"Directory not found: {path}")

# Image size and batch size
image_size = (180, 180)
batch_size = 32

# Advanced Image Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.3),
    layers.RandomHeight(0.2),
    layers.RandomWidth(0.2),
    keras_cv.layers.RandomShear(x_factor=0.3, y_factor=0.3),  # ✅ Corrected
])

# Load datasets efficiently
AUTOTUNE = tf.data.AUTOTUNE
train_ds = keras.utils.image_dataset_from_directory(
    data_train_path, image_size=image_size, batch_size=batch_size, label_mode='categorical', shuffle=True
).cache().prefetch(buffer_size=AUTOTUNE)

val_ds = keras.utils.image_dataset_from_directory(
    data_valid_path, image_size=image_size, batch_size=batch_size, label_mode='categorical'
).cache().prefetch(buffer_size=AUTOTUNE)

# Compute class weights to handle class imbalance
class_counts = np.zeros(3)  # Number of classes: melanoma, nevus, seborrheic keratosis
for images, labels in train_ds:
    class_counts += np.sum(labels.numpy(), axis=0)

total_images = np.sum(class_counts)
class_weights = {i: total_images / (count + 1e-6) for i, count in enumerate(class_counts)}  # ✅ Prevent division by zero

# Model Definition
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)  # Normalize pixel values

    # Load ResNet50 (Frozen initially)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(180, 180, 3))
    base_model.trainable = False  # Freeze base model initially
    x = base_model(x, training=False)  # Set to inference mode
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)  # Normalize activations
    x = layers.Dropout(0.3)(x)  # Reduce overfitting

    outputs = layers.Dense(num_classes, activation="softmax", dtype='float16')(x)  # ✅ Ensure correct dtype
    return keras.Model(inputs, outputs)

input_shape = image_size + (3,)
num_classes = 3  # melanoma, nevus, seborrheic keratosis

# Initialize and compile model
model = make_model(input_shape=input_shape, num_classes=num_classes)
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Train top layers first
epochs = 10
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping],
    class_weight=class_weights
)

# Unfreeze the base model for fine-tuning
base_model = model.get_layer("resnet50")  # ✅ Correct retrieval
base_model.trainable = True  # ✅ Unfreeze
optimizer = keras.optimizers.Adam(learning_rate=1e-5)  # Lower LR for fine-tuning
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue fine-tuning
epochs_finetune = 20
history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs_finetune,
    callbacks=[early_stopping],
    class_weight=class_weights
)

# Evaluate the model
test_ds = keras.utils.image_dataset_from_directory(
    data_test_path,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
).cache().prefetch(buffer_size=AUTOTUNE)

test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {test_accuracy:.2f}")

# Convert model to TensorFlow Lite for deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# ✅ Enable TF Select to handle unsupported ops
converter.allow_custom_ops = True  # Allow unsupported TensorFlow ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Use built-in TFLite operations
    tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops fallback
]

# ✅ Enable full model optimization for smaller size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
with open("melanoma_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TFLite with TF Select and saved!")

# Plot accuracy
train_acc = history.history['accuracy']
train_acc.extend(history_finetune.history['accuracy'])

val_acc = history.history['val_accuracy']
val_acc.extend(history_finetune.history['val_accuracy'])

plt.figure(figsize=(8, 6))
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
