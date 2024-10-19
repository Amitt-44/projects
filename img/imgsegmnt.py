import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained Pix2Pix model
model_path = r'C:\Users\44ami\Desktop\ML projects\img\my_pix2pix_model.h5'  # Use a raw string

model = load_model(model_path)
model = load_model(model_path, compile=False)


# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to the input size of the model
    image = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Function to post-process the output
def postprocess_output(output):
    output = (output[0] * 255).astype(np.uint8)  # Convert back to pixel values
    return Image.fromarray(output)

# Streamlit app layout
st.title("Pix2Pix Image Segmentation")
st.write("Upload an image to segment it using the Pix2Pix model.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    input_image = preprocess_image(original_image)

    # Predict using the model
    st.write("Segmenting the image...")
    with st.spinner('Please wait...'):
        output_image = model.predict(input_image)
    
    # Post-process the output
    segmented_image = postprocess_output(output_image)

    # Display segmented image
    st.image(segmented_image, caption='Segmented Image', use_column_width=True)

# Footer
st.write("### Instructions:")
st.write("1. Upload an image file (jpg, jpeg, or png).")
st.write("2. Wait for the segmentation result.")
