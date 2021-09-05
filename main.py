import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras

class_names = ['Bicycle', 'Boat', 'Cat', 'Motorbike', 'People', 'Table']
img_height = 180
img_width = 180

st.write('''<style>
            body{
            text-align:center;
            background-color:#ACDDDE;
            }
            </style>''', unsafe_allow_html=True)

st.title('Image Classifier')

#loading model

model = keras.models.load_model('./models/model/')

file_type = 'jpg'

st.text("The app can classify images among the 6 classes:\nBicycle, Boat, Cat, Motorbike, People, Table")

uploaded_file = st.file_uploader("Choose a  file",type = file_type)

if uploaded_file != None:

    image = Image.open(uploaded_file)

    image = image.filter(ImageFilter.MedianFilter)
    print(image.size)
    image = image.resize((180, 180))
    st.image(image)
    
    img_array = keras.preprocessing.image.img_to_array(image)
    print(img_array.shape)
    np.reshape(img_array,(180,180,3))
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    st.text("Class: {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))




