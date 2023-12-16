import streamlit as st
import numpy as np
import pandas as pd
import cv2
from model_architecture import build_model

def load_model():
    model = build_model()
    model.load_weights('model2.h5')
    return model
model = load_model()
st.title('WATER MELON RIPENESS DETECTION APP')
uploaded_file =st.file_uploader(label = 'Upload an image..', type=['jpg', 'jpeg', 'png'])

def classify_image(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (75, 75))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype='uint8')
    opencv_image = cv2.imdecode(file_bytes, flags=1)
    img = cv2.resize(opencv_image, (200, 200))
    st.image(img, channels='BGR')
    prediction = classify_image(opencv_image, model)
    if prediction[0][0] > 0.5:
        st.write('Ripe Watermelon')
    else:
        st.write('Unripe Watermelon')