import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('waste_classifier.h5')
class_names = ['metal', 'paper', 'plastic']

st.title('Waste Sorting Classifier')
st.write('Upload an image to classify it as metal, paper or plastic')

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
