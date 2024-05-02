### Multi-Class Classification for Chest X-Ray Images
### Description:
This project focuses on developing a deep learning model capable of classifying Chest X-Ray images into three categories: COVID, Pneumonia, or No Disease. The model utilizes convolutional neural networks (CNNs) to extract features from the images and predict the respective classes.

#### Data Preprocessing:
- Chest X-Ray images are converted to RGB channels and resized to dimensions of 128 x 128 pixels.
- Separate folders are created for each class to organize the dataset.
- Data is split into training, testing, and validation sets using the `train_test_split` library.
- Pixel values are normalized, and data is loaded into data loaders for efficient processing.

#### Model Architecture:
- The model architecture is built using the `Sequential()` API in Keras.
- Multiple convolutional layers followed by max-pooling layers are used for feature extraction and down-sampling.
- The final output of the max-pooling layer is flattened to a single dimension.
- Relu activation function is used for the hidden layers.
- Softmax activation function is applied to the output dense layer for multi-class classification.

#### Training:
- The model is trained for 150 epochs using the Adam optimizer.
- Hyperparameters such as epochs, batch size, and learning rate are fine-tuned to optimize performance.
- Dropout layers and early stopping mechanisms are implemented to prevent overfitting.

#### Evaluation Metrics:
- Model performance is evaluated using metrics such as accuracy, confusion matrix, precision, and recall.
- The model demonstrates good performance on both training and validation datasets, indicating robustness and generalization capability.

#### Difficulties:
1. Fine-tuning the model parameters to achieve better results.
2. Preventing overfitting by implementing dropout layers and early stopping mechanisms.
