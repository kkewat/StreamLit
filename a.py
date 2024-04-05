import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Load your trained model
model = tf.keras.models.load_model("C:\\Users\\Lenovo\\Downloads\\resnet50_model_multilabel.keras")

def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img = image.img_to_array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def classes(predicted_class):
    class_mapping = {
        0: 'Apple_Leaf_Healthy',
        1: 'Apple_Rust_Leaf',
        2: 'Apple_Scab_Leaf',
        # Add other class mappings here
    }
    predicted_class_name = class_mapping.get(predicted_class, 'Unknown Class')
    return predicted_class_name

def predict(image_file):
    img = preprocess_image(image_file)
    preds = model.predict(img)
    predicted_class = np.argmax(preds)
    predicted_class_name = classes(predicted_class)
    return predicted_class_name

def main():
    st.title("Image Classifier")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        predicted_class = predict(uploaded_file)
        st.write("Prediction:", predicted_class)

if __name__ == "__main__":
    main()
