from pydoc import classname
import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
model = load_model("model_120.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def predict_image(image):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    prediction_percentages = prediction[0]
    sorted_indices = np.argsort(prediction_percentages)[::-1]
    
    class_names_sorted = [class_names[i][2:] for i in sorted_indices]
    confidence_scores_sorted = [prediction_percentages[i] for i in sorted_indices]
    confidence_percentages_sorted = [round(score * 100, 2) for score in confidence_scores_sorted]
    
    # Displaying the graph
    fig, ax = plt.subplots()
    ax.bar(class_names_sorted, confidence_percentages_sorted)
    ax.set_ylabel('Confidence Percentage')
    ax.set_xlabel('Class')
    ax.set_title('Class Confidence Scores')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    for i in sorted_indices:
        class_name = class_names[i]
        confidence_score = prediction_percentages[i]
        confidence_percentage = round(confidence_score * 100, 2)
        st.write("Class:", class_name[2:], "--> Confidence Score:", confidence_percentage)
    
    return class_names_sorted[0], confidence_scores_sorted[0]



st.title("Anomaly Classification")
st.write("Upload an image to classify")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True, width=300)
    st.write("")

    st.write("Classifying...")
    class_name, confidence_score = predict_image(image)
    st.write("Class:", class_name)
    st.write("Confidence Score:", confidence_score)
