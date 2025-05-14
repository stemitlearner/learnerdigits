import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2

# Load the trained model
model = tf.keras.models.load_model('mnist_digit_classifier.h5')

st.title("MNIST Digit Recognition App")
st.markdown("Draw a digit (0-9) below and click **Predict** to see the result.")

# Canvas for drawing
canvas_result = st.canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data

    if st.button("Predict"):
        # Convert to grayscale and resize to 28x28
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0  # Normalize

        # Invert colors because MNIST is white on black
        img = 1 - img

        # Expand dimensions to match model input
        img_input = np.expand_dims(img, axis=(0, -1))
        img_input = img_input.reshape(1, 28, 28)

        # Predict
        prediction = model.predict(img_input)
        predicted_digit = np.argmax(prediction)

        st.write(f"### Predicted Digit: **{predicted_digit}**")
        st.bar_chart(prediction[0])

