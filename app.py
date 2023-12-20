import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from preprocess import cascaded_preprocessing  
import streamlit as st

class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']  # Defining the class labels for brain tumor types

model = tf.keras.models.load_model("effnet.h5")  # Load the pre-trained brain tumor classification model

@st.cache_data
def process_image(file_buffer):
    image = np.frombuffer(file_buffer.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
def main():
    st.title("Brain Tumor Detection")

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        cv_image = process_image(uploaded_file)

        st.image(cv_image, caption="Uploaded Image", use_column_width=True)
        image_size = 224
        resized_image = cv2.resize(cv_image, (image_size, image_size))
        preprocessed_image = cascaded_preprocessing(resized_image)

        prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))

        predicted_class = class_labels[np.argmax(prediction)]  # Fetching the predicted class based on the highest probability
        st.info(f'Predicted Tumor Type: {predicted_class}')  # Display the predicted tumor type

if __name__ == "__main__":
    main()


# # python -m streamlit run app.py --server.port 8080