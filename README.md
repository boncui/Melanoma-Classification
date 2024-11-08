# Melanoma-Classification

**Collaborators:** Israel Chavez, David Cui

## Goal

We came together to work on a project for graduate school and for our portfolios. As students studying CS and Math, we wanted to apply our learning to the real world with the intent to make a difference in society. Due to our goals, we decided to pick a dataset involved in medicine because of its applicational nature.

## Dataset Introduction

**Link:** [Melanoma Dataset from Kaggle](https://www.kaggle.com/datasets/wanderdust/skin-lesion-analysis-toward-melanoma-detection/data)

> "The goal is to support research and development of algorithms for automated diagnosis of melanoma, the most lethal skin cancer" (Noel C.F. Codella et al 2017).

The dataset has 2750 samples of three different skin diseases: Melanoma, Nevus, and Seborrheic Keratosis.

## Brainstorming

To begin the project, we first looked into the dataset to inspect the samples. We noticed a heavy class imbalance in each of the train, validation, and testing directories, as there were not an equal number of images for each of the skin diseases. To address this, we knew we had to implement a class balancing function in the preprocessing step.

We also looked into similar projects on GitHub and Kaggle as references to learn what worked best for other projects and to avoid reinventing the wheel. To classify images, Convolutional Neural Networks (CNNs) are a popular choice, as CNNs are specialized artificial neural networks particularly effective for tasks involving image recognition, object detection, and similar types of pattern recognition. For the project, we used **Keras, TensorFlow, Numpy,** and **Matplotlib**.

## Process

1. **Set up environment and load dataset**
2. **Data augmentation** - We applied the following data augmentation:

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomWidth(0.2),
    layers.RandomHeight(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
])
```

3. Preprocessing and PreFetching - The training and validation datasets are loaded from directories using the image_dataset_from_directory function, which automatically labels the images based on their folder names. The data is set to be in categorical format (one-hot encoded), and prefetching is used to optimize performance.

4. Calculating Weight Classes - Class weights are calculated to handle class imbalance by adjusting the importance of each class during training. This is achieved by counting the instances in each class and calculating weights to give more importance to underrepresented classes.

5. Data Visualization Preprocessing Step - This section visualizes the effect of augmentation by displaying original and augmented images side by side.

6. Defining the Model - The model is created using a pre-trained ResNet50 as the base, allowing it to leverage features learned from a large dataset (ImageNet). The data is normalized, and the ResNet50 model is loaded with include_top=False, meaning it excludes its original classification head so that the model can be adapted for this specific task. A global average pooling layer and a dropout layer (rate 0.3) help prevent overfitting. Finally, a dense layer with softmax activation is used for the classification output.

7. Model Compiling - The model is compiled with the Adam optimizer (learning rate of 0.0001) and categorical cross-entropy loss, appropriate for multi-class classification.

8. Model Training - The model is trained for up to 50 epochs, with early stopping based on validation loss to avoid overfitting. The patience parameter is set to 12, meaning the training will stop if the validation loss does not improve for 12 consecutive epochs. Class weights are applied to handle class imbalance.

9. Model Testing - The test dataset is loaded, preprocessed, and evaluated to obtain the model’s performance on unseen data, providing metrics like test loss and accuracy.

## Results

Our final model was based on a ResNet50 architecture with the following settings and hyperparameters:

# Optimizer: Adam

Data Augmentation:

```python
layers.RandomFlip("horizontal")
layers.RandomRotation(0.2)
layers.RandomZoom(0.2)
layers.RandomWidth(0.2)
layers.RandomHeight(0.2)
layers.RandomBrightness(0.2)
layers.RandomContrast(0.2)
```

### Network Architecture

- **Rescaling Layer**: Normalizes pixel values to a [0, 1] range.
- **Pre-trained Base Model**: ResNet50 with ImageNet weights (top layer excluded), set to be trainable.
- **Global Average Pooling**: Summarizes features spatially.
- **Dropout Layer**: Drops 30% of the neurons at that layer to prevent overfitting.
- **Output Layer**: Dense layer with softmax activation for classification into 3 classes.

### Training Details

- **Epochs**: 50
- **Early Stopping**: Patience of 12 epochs, monitoring `val_loss`, with `restore_best_weights=True`.
- **Class Weighting**: Used to counter class imbalance.

### Final Results

- **Test Accuracy**: 0.71
- **Early Stopping**: Did not activate; all 50 epochs ran
- **Best Epoch**: Epoch 47
- **Training Accuracy**: 0.9543
- **Validation Accuracy**: 0.7533
- **Validation Loss**: 0.8276 at Epoch 40 (minimum)

### Potential Improvements

- Adjust early stopping to monitor `val_accuracy` instead of `val_loss` with a higher patience (e.g., 20 epochs) to better optimize validation accuracy.
- Increase the number of epochs by 10-30.
