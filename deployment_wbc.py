#!/usr/bin/env python
# coding: utf-8

# In[1]:

#custom web apps for machine learning and data science
import streamlit as st
import tensorflow.keras as keras
from PIL import Image
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np

import keras.utils as image

models = keras.models.load_model("cnn_27.h5") # load model

#page configuration of the Streamlit app
#specifies the title of the web page
#specifies the icon of the page
st.set_page_config(
    page_title="wbc Image Classifier",
    page_icon=":art:",
    layout="wide",
    initial_sidebar_state="expanded"
)

target_size = (60, 60)
# allows the user to upload an image file
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:

#opened using the PIL library's Image.open function
#resized to the target size
#converted to a numpy array
#batch dimension is added most machine learning models expect input data to have a batch dimension

    image = Image.open(uploaded_file)
    image = image.resize(target_size)
    image_array = np.array(image)
    
    image_array = np.expand_dims(image_array, axis=0)
    image_array =  image_array/255 
    
# which returns an array of probabilities for each class
#class with the highest probability

#predicted class is displayed to the user

    y_predict=np.argmax(models.predict(image_array))
    y_predict
    if y_predict==0:
        st.write("EOSINOPHIL")
        
    elif y_predict==1:
         st.write("LYMPHOCYTE")
    elif y_predict==2:
         st.write("MONOCYTE")
    else:
         st.write("NEUTROPHIL")

    st.image(image, caption=f'Uploaded WBC Image prediction ({y_predict})', width = 200)


# In[ ]:




