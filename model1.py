import numpy as np
import pandas as pd
import os
from IPython.display import display
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_files
from tqdm import tqdm
from collections import Counter

# Check if the local dataset directory exists
print(os.listdir("skin-lesions"))


data_train_path = 'skin-lesions/train'
data_valid_path = 'skin-lesions/valid'
data_test_path = 'skin-lesions/test'

for path in [data_train_path, data_valid_path, data_test_path]:
    if not os.path.exists(path):
        print(f"Directory not found: {path}")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
    rotation_range=20,   # Randomly rotate images
    width_shift_range=0.2, # Randomly shift images horizontally
    height_shift_range=0.2, # Randomly shift images vertically
    horizontal_flip=True  # Randomly flip images
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Only rescale for validation
test_datagen = ImageDataGenerator(rescale=1.0 / 255)   # Only rescale for testing

train_ds = train_datagen.flow_from_directory(
    data_train_path,
    target_size=(180, 180),  # Resize images to the specified size
    batch_size=32,            # Number of images per batch
    class_mode='categorical'  # Use 'categorical' for multi-class classification
)


valid_ds = valid_datagen.flow_from_directory(
    data_valid_path,
    target_size=(180, 180),
    batch_size=32,
    class_mode='categorical'
)


test_ds = test_datagen.flow_from_directory(
    data_test_path,
    target_size=(180, 180),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Do not shuffle for test data to maintain order
)