import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.data import AUTOTUNE
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# Paths
data_train_path = 'skin-lesions/train'
data_valid_path = 'skin-lesions/valid'
data_test_path = 'skin-lesions/test'

# Checking if directories exist
for path in [data_train_path, data_valid_path, data_test_path]:
    if not os.path.exists(path):
        print(f"Directory not found: {path}")

# Image settings
image_size = (180, 180)
batch_size = 32

# 🔥 Enhanced Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomWidth(0.2),
    layers.RandomHeight(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.GaussianNoise(0.1),  # 🚀 Adds Gaussian noise for robustness
])

# Load datasets efficiently
train_ds = keras.utils.image_dataset_from_directory(
    data_train_path, image_size=image_size, batch_size=batch_size, label_mode='categorical'
)

val_ds = keras.utils.image_dataset_from_directory(
    data_valid_path, image_size=image_size, batch_size=batch_size, label_mode='categorical'
)

# Compute class weights for imbalance handling
class_counts = np.zeros(3)
for images, labels in train_ds:
    class_counts += np.sum(labels.numpy(), axis=0)

total_images = np.sum(class_counts)
class_weights = {i: total_images / (count + 1e-6) for i, count in enumerate(class_counts)}

# Model Definition with Stronger Regularization
def make_model(input_shape, num_classes, trainable=False):
    inputs = keras.Input(shape=input_shape)
    
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    # Load ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(180, 180, 3))
    base_model.trainable = trainable  # 🔥 Initially freeze ResNet50
    x = base_model(x, training=False)

    # 🔥 Add Batch Normalization
    x = layers.BatchNormalization()(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # 🔥 Apply Stronger Regularization (L2 Weight Decay + Dropout)
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)  # Increased dropout rate

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base_model  # 🔥 Return both the model and the base_model

# Set input shape and number of classes
input_shape = image_size + (3,)
num_classes = 3

#correcy learning rate
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=1000, alpha=0.1
)

# Instantiate optimizer (fix: convert lr_schedule to float initially)
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

"""ALTERNATIVE"""
# optimizer = keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.CosineDecay(
#     initial_learning_rate=1e-3, decay_steps=1000, alpha=0.1
# ))


# Compile model (with frozen ResNet)
model, base_model = make_model(input_shape=input_shape, num_classes=num_classes, trainable=False)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks: EarlyStopping + ReduceLROnPlateau
callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)  # 🚀 Adjust LR dynamically
]

# 🚀 Train top layers first (ResNet frozen)
history = model.fit(
    train_ds, validation_data=val_ds, epochs=25, callbacks=callbacks, class_weight=class_weights
)

# 🔥 Gradual Unfreezing of ResNet50 for Fine-Tuning
for layer in base_model.layers[-30:]:  # Only unfreeze last 25 layers
    layer.trainable = True
fine_tune_epochs = 30

# Recompile the model for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🚀 Train with Fine-Tuning
history_finetune = model.fit(
    train_ds, validation_data=val_ds, epochs=fine_tune_epochs, callbacks=callbacks, class_weight=class_weights
)

# Evaluate on test dataset
test_ds = keras.utils.image_dataset_from_directory(
    data_test_path, image_size=image_size, batch_size=batch_size, label_mode='categorical'
).prefetch(buffer_size=AUTOTUNE)

test_loss, test_accuracy = model.evaluate(test_ds)
print(f"🔥 Final Test Accuracy: {test_accuracy:.2f}")

# Plot Accuracy
plt.figure(figsize=(8, 6))
train_acc = history.history['accuracy'] + history_finetune.history['accuracy']
val_acc = history.history['val_accuracy'] + history_finetune.history['val_accuracy']
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
