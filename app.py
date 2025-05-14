import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load your modern saved model
model = tf.keras.models.load_model('mnist_digit_classifier.keras')

st.title("MNIST Digit Recognition App")
st.markdown("Draw a digit (0-9) below and click **Predict** to see the result.")

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
        img_pil = Image.fromarray((img[:, :, 0:3] * 255).astype(np.uint8))
        img_pil = img_pil.convert("L")
        img_pil = img_pil.resize((28, 28))
        img_array = np.asarray(img_pil) / 255.0
        img_array = 1 - img_array  # Invert colors

        img_array = img_array.reshape(1, 28, 28)

        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        st.write(f"### Predicted Digit: **{predicted_digit}**")
        st.bar_chart(prediction[0])
