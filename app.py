import streamlit as st
import keras
from keras import layers
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import zipfile
import requests

st.title('Waste Sorting Classifier')
st.write('Upload an image to classify it as metal, paper or plastic')

@st.cache_resource
def load_model():
    model = keras.Sequential([
        keras.Input(shape=(150, 150, 3)),
        layers.Rescaling(1./255),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    return model

model = load_model()
class_names = ['metal', 'paper', 'plastic']

uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img = image.resize((150, 150))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    st.write(f'This is: **{predicted_class}**')
    st.write(f'Confidence: **{confidence}%**')
   
