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
image_size = (224, 224)
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
test_ds = keras.utils.image_dataset_from_directory(
    data_test_path,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)


# Preteching images to improve performance
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# The number of classes - melanoma, nevus, seborrheic keratosis
class_counts = np.zeros(3)
for images, labels in train_ds:
    class_counts += np.sum(labels.numpy(), axis=0)
total_images = np.sum(class_counts)
class_weights = {i: total_images / (count + 1e-6) for i, count in enumerate(class_counts)}

print(f"Class weights: {class_weights}")        # The class weights are used to balance the dataset (to counteract the class imbalance)

# Model Selection
    #Restnet50
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    # ResNet50 base
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False #Start w frozen resnet
     
     
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    #apply stronger regularization
    x = layers.BatchNormalization()(x)  # Normalize activations
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)  # L2 Regularization
    x = layers.Dropout(0.2)(x)


    # Classification layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    return model, base_model

# Setting input shape and number of classes
input_shape = image_size + (3,)
num_classes = 3 

model, base_model = make_model(input_shape=input_shape, num_classes=num_classes)

# optimizer = keras.optimizers.Adam(learning_rate=0.0001)
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0005, decay_steps=1000, alpha=0.1
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

history = model.fit(
    train_ds, validation_data=val_ds, epochs=30, callbacks=[early_stopping], class_weight=class_weights
)

# Unfreeze the last few layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Fine-tune only last 10 layers
    layer.trainable = False

# Compile again with a lower learning rate for fine-tuning
optimizer = keras.optimizers.Adam(learning_rate=1e-5) 
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define a new Early Stopping Callback for Fine-Tuning
early_stopping_fine = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 🏋️ Train Again with Fine-Tuning
fine_tune_epochs = 15  # Fine-tune for 10-15 more epochs
history_fine = model.fit(
    train_ds, validation_data=val_ds, epochs=fine_tune_epochs, callbacks=[early_stopping], class_weight=class_weights
)

model.summary()

# Evaluating the model on the test dataset

test_loss, test_accuracy = model.evaluate(test_ds)
print(f"🚀 Final Test Accuracy: {test_accuracy:.2f}")

# ✅ Save the trained model
model.save("melanoma_classifier.keras")  # TensorFlow format

# ✅ Verify the saved model
print("\n✅ Verifying saved model...")
loaded_model = tf.keras.models.load_model("melanoma_classifier.keras")
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


