import streamlit as st
from tensorflow import keras
from keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
from PIL import Image # Streamlit work very easily for images
import cv2 # for decoding image
from keras.applications.mobilenet_v2 import preprocess_input,decode_predictions

@st.cache()
def load_model():
	model = MobileNetV2()
	return model

st.title('Image - Classifier')

# to upload image
upload = st.sidebar.file_uploader('Label=Upload the Image')

# Following lines of code 
# reads image and decode it in csv file
if upload is not None:  
  file_bytes = np.asarray(bytearray(upload.read()),dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes,1)
  # this code for converting BGR to RGB
  opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)
  img = Image.open(upload)
  st.image(img,caption = 'Uploaded Image',width=300)
  model = load_model()	
  if st.sidebar.button('Predict'):
    st.sidebar.write("Result : ")
    # Now taking that image and use it in our model
    x = cv2.resize(opencv_image,(224,224))
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    y = model.predict(x)
    label = decode_predictions(y)
    for i in range(3):
      out = label[0][i]
      st.sidebar.title('%s (%.2f%%)' % (out[1], out[2]*100))
