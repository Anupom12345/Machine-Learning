import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image 
import glob 
import os

# Define the model file path
model_path = 'ml_model_extended.keras' # Changed filename to avoid overwriting the old model

# Check if the model file already exists
if os.path.exists(model_path):
    print("Found existing model file. Loading the model...")
    # Load the saved model
    loaded_model = tf.keras.models.load_model(model_path)
else:
    print("No model file found. Training a new, more complex model...")
    # Load the MNIST dataset (handwritten digits 0–9)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values to range [0, 1] for better model performance
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Reshape data to include a channel dimension (needed for Conv2D)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Build the more complex CNN model with additional layers
    model = tf.keras.Sequential([
        # First Block
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D((2,2)),

        # Second Block (New)
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),

        # Third Block (New)
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),

        # Flatten layer
        tf.keras.layers.Flatten(),
        
        # Dense layers
        tf.keras.layers.Dense(128, activation='relu'), # Increased neurons for complexity
        tf.keras.layers.Dropout(0.5), # Add a dropout layer to prevent overfitting (New)
        tf.keras.layers.Dense(64, activation='relu'), # Added a new dense layer
        
        # Output layer
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model with optimizer, loss function, and metrics
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Print a summary of the new model architecture
    model.summary()

    # Train the model for 10 epochs (increased epochs for better convergence)
    model.fit(x_train, y_train, epochs=10)

    # Evaluate the model on the test dataset
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Pick a random test image and predict
    index = np.random.randint(0, len(x_test))
    sample_image = x_test[index]
    true_label = y_test[index]
    prediction = model.predict(sample_image.reshape(1, 28, 28, 1))
    predicted_label = np.argmax(prediction)

    # Show the image with true and predicted labels
    plt.imshow(sample_image.squeeze(), cmap='gray')
    plt.title(f"True: {true_label}, Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

    # Save the trained model to a file
    model.save(model_path)
    loaded_model = model

# From here, the code uses the `loaded_model` variable,
# regardless of whether it was trained or loaded.

# Get all image file paths from "mnist_samples" folder
image_files = glob.glob("mnist_samples/*.png")

# Loop through each image and make a prediction
for image_path in image_files:
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    pred = loaded_model.predict(img_array)
    predicted_digit = np.argmax(pred)

    # Show the image and predicted result
    plt.imshow(img_array.squeeze(), cmap='gray')
    plt.title(f"{image_path} -> Predicted: {predicted_digit}")
    plt.axis('off')
    plt.show()