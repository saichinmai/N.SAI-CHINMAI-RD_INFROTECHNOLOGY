import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np

# Load trained model
model = tf.keras.models.load_model("digit_model.h5")

st.set_page_config(page_title="Digit Recognition")

st.title("AI Handwritten Digit Recognition")
st.write("Draw a digit (0–9)")

# Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    width=300,
    height=300,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Predict"):

    if canvas_result.image_data is not None:

        image = canvas_result.image_data.astype("uint8")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

        # Threshold
        _, thresh = cv2.threshold(
            gray,
            50,
            255,
            cv2.THRESH_BINARY
        )

        # Find digit area
        coords = cv2.findNonZero(thresh)

        if coords is not None:

            x, y, w, h = cv2.boundingRect(coords)

            digit = thresh[y:y+h, x:x+w]

            # Resize to 20x20
            digit = cv2.resize(digit, (20, 20))

            # Create blank 28x28 image
            final_img = np.zeros((28, 28), dtype=np.uint8)

            # Center digit
            final_img[4:24, 4:24] = digit

            # Normalize
            final_img = final_img / 255.0

            # Reshape for CNN
            input_img = final_img.reshape(
                1, 28, 28, 1
            )

            # Prediction
            prediction = model.predict(input_img)

            predicted_digit = np.argmax(prediction)

            confidence = np.max(prediction) * 100

            st.success(
                f"Predicted Digit: {predicted_digit}"
            )

            st.write(
                f"Confidence: {confidence:.2f}%"
            )

if st.button("Clear"):
    st.rerun()