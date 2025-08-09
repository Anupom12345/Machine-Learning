# Handwritten Digit Recognition with a Convolutional Neural Network (CNN)

This project demonstrates how to build, train, and use a **Convolutional Neural Network (CNN)** to recognize handwritten digits. It uses the popular **MNIST dataset**, a collection of 60,000 training and 10,000 testing images of handwritten digits from 0 to 9. The script also includes a section to test the trained model on your own custom images.

---

### **Project Goal**

The primary goal of this project is to create a robust machine learning model that can accurately classify handwritten digits from images. The model is a simple CNN, a type of neural network particularly effective for image-based tasks.

---

## **How It Works**

The Python script performs the following key steps:

1.  **Data Loading and Preprocessing**: The MNIST dataset is loaded directly from TensorFlow. The pixel values of the images are **normalized** to a range of [0, 1] to improve model performance, and the images are reshaped to the correct format for the CNN.
2.  **Model Building**: A sequential CNN model is constructed using TensorFlow's Keras API. It consists of:
    * **Convolutional Layer (`Conv2D`)**: Extracts features from the images, like edges and curves.
    * **Pooling Layer (`MaxPooling2D`)**: Reduces the spatial size of the feature maps, which helps to reduce computation and prevent overfitting.
    * **Flatten Layer**: Converts the 2D feature maps into a 1D vector to be fed into the fully connected layers.
    * **Dense Layers**: Standard neural network layers that perform the final classification. The output layer uses a **softmax** activation function to output probabilities for each of the 10 digit classes.
3.  **Training**: The model is trained on the training data for 5 **epochs**, which means the entire dataset is passed forward and backward through the neural network five times.
4.  **Evaluation and Prediction**: After training, the model's accuracy is evaluated on the test dataset. A random test image is selected and the model's prediction is displayed alongside the true label.
5.  **Model Persistence**: The trained model is saved to a file (`ml_model.keras`) so it can be loaded later without retraining.
6.  **Custom Image Prediction**: The script loads the saved model and uses it to predict the digits in a folder of custom images named `test_images`. It processes each image by converting it to grayscale, resizing it, and normalizing its pixel values before making a prediction.

---

## **Setup and Installation**

### **Prerequisites**

You'll need Python and the following libraries installed. You can install them using `pip`:

```bash
pip install tensorflow numpy matplotlib pillow
````

### **Running the Script**

1.  **Save the code**: Save the provided Python code as a file named `main.py`.

2.  **Create a folder for custom images**: Create a new folder in the same directory as your script and name it `test_images`.

3.  **Add custom images (optional)**: Place your own handwritten digit images (in `.png` or `.jpg` format) into the `test_images` folder. The script will automatically resize them to 28x28 pixels.

4.  **Run the script**: Execute the script from your terminal:

    ```bash
    python main.py
    ```

The script will download the MNIST dataset, train the model, print the test accuracy, display a random test image with its prediction, and then show the predictions for all images in the `test_images` folder.

```
```