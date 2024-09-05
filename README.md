## Digit Recognizer

This project is focused on building a digit recognition system using a Convolutional Neural Network (CNN). The model is trained on the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9). The goal of the project is to classify the digits with high accuracy.

## Project Overview

- **Dataset:** The MNIST dataset contains 60,000 training images and 10,000 test images, each labeled with the corresponding digit (0-9).
- **Model:** A CNN model is implemented to classify the images. The architecture includes convolutional layers, pooling layers, and dense layers.
- **Evaluation:** The model's performance is evaluated using accuracy and confusion matrix metrics on the test dataset.

## Project Steps

1. **Data Loading and Preprocessing:**
   - Load the MNIST dataset.
   - Normalize the pixel values to the range [0, 1].
   - Reshape the images to match the input shape required by the CNN.
   
2. **Model Architecture:**
   - The CNN model consists of several convolutional layers followed by max-pooling and dropout layers.
   - Fully connected layers are added at the end, with a softmax activation function for classification.

3. **Model Training:**
   - The model is trained using the Adam optimizer and categorical cross-entropy loss function.
   - The training process includes multiple epochs, with accuracy and loss tracked.

4. **Model Evaluation:**
   - Evaluate the model on the test dataset.
   - Generate a classification report, including precision, recall, and F1-score for each digit.
   - Display the confusion matrix to analyze misclassifications.

## Dependencies

- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- Seaborn

You can install the required libraries using:

```bash
pip install tensorflow numpy matplotlib seaborn
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/digit-recognizer.git
```

2. Navigate to the project directory:

```bash
cd digit-recognizer
```

3. Run the Jupyter Notebook:

```bash
jupyter notebook Digit_recognizer.ipynb
```

4. Follow the steps in the notebook to train and evaluate the model.

## Dataset

The MNIST dataset is used in this project. It is automatically downloaded when using TensorFlow/Keras, but you can also manually download it if needed.

## Results

The CNN model achieves an accuracy of around 97% on the test dataset. The model is capable of recognizing digits with high precision and recall across all classes.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request.

