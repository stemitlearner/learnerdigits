import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load the trained model
model = tf.keras.models.load_model('mnist_digit_classifier (1).h5')

st.title("MNIST Digit Recognition App")
st.markdown("Draw a digit (0-9) below and click **Predict** to see the result.")

# Create a canvas component
canvas_result = st_canvas(
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
        # Convert to grayscale and resize
        img_pil = Image.fromarray((img[:, :, 0:3] * 255).astype(np.uint8))
        img_pil = img_pil.convert("L")  # Convert to grayscale
        img_pil = img_pil.resize((28, 28))
        img_array = np.asarray(img_pil) / 255.0
        img_array = 1 - img_array  # Invert to match MNIST

        # Prepare for model
        img_array = img_array.reshape(1, 28, 28)

        # Predict
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        st.write(f"### Predicted Digit: **{predicted_digit}**")
        st.bar_chart(prediction[0])
