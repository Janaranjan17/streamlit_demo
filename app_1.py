import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
model = load_model("model_final.h5")

# Define the class labels
verbose_name = {
    0: "Non Demented",
    1: "Very Mild Demented",
    2: "Mild Demented",
    3: "Moderate Demented",
}

def predict_label(img):
    # Convert to grayscale and resize
    test_image = img.convert("L").resize((128, 128))
    test_image = image.img_to_array(test_image) / 255.0
    test_image = test_image.reshape(-1, 128, 128, 1)

    # Prediction
    predict_x = model.predict(test_image)
    classes_x = np.argmax(predict_x, axis=1)

    return verbose_name[classes_x[0]]

# Streamlit UI
st.title("Dementia Stage Classification")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_disp = Image.open(uploaded_file)
    st.image(image_disp, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        prediction = predict_label(image_disp)
        st.success(f"Prediction: {prediction}")
