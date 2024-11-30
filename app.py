import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load your pre-trained model
model = load_model("sketchrnn_model.h5")
def load_class_names(file_path):
    with open(file_path, "r") as file:
        return [line.strip() for line in file]

class_names = load_class_names("class_names.txt")

def preprocess_image(image):
    """
    Prepares the input image for the model:
    - Resize it to the model's expected dimensions.
    - Normalize pixel values to [0, 1].
    """
    size = (28, 28)  # Adjust based on your model's input size
    image = ImageOps.grayscale(image)  # Convert to grayscale if needed
    image = image.resize(size)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return image_array.reshape(1, *size, 1)  # Add batch and channel dimensions

# Streamlit app
st.title("Sketch Classifier")
st.write("Draw something on the canvas below and let the model predict its class!")

# Create a canvas for drawing
canvas_result = st.canvas(
    stroke_width=5,  # Adjust pen thickness
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Convert the canvas drawing to a PIL image
    img = Image.fromarray((canvas_result.image_data[:, :, 3] * 255).astype("uint8"))
    
    # Preprocess the image for the model
    input_image = preprocess_image(img)

    # Get predictions from the model
    predictions = model.predict(input_image)
    top_k = tf.nn.top_k(predictions, k=5)  # Get top-5 predictions

    # Display the predictions
    st.write("## Predictions")
    for i in range(5):
        class_index = top_k.indices[0, i].numpy()  # Get the class index
        class_name = class_names[class_index]  # Map the index to the class name
        probability = top_k.values[0, i].numpy() * 100  # Convert to percentage
        st.write(f"{i + 1}. {class_name}: {probability:.2f}%")
