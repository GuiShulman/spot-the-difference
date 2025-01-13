import streamlit as st
import cv2
import numpy as np
from PIL import Image

def process_image(image):
    # Convert to numpy array
    image_np = np.array(image)

    # Split the image into two halves (horizontal split)
    height, width, _ = image_np.shape
    half_height = height // 2
    top_half = image_np[:half_height, :]
    bottom_half = image_np[half_height:, :]

    # Convert to grayscale
    top_gray = cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY)
    bottom_gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference
    difference = cv2.absdiff(top_gray, bottom_gray)

    # Threshold the difference to identify significant changes
    _, thresholded_diff = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    # Find contours of the differences
    contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around differences on the original image
    output_image = image_np.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Ignore small differences
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output_image, thresholded_diff

# Streamlit App
st.title("Spot the Difference")
st.write("Upload an image where the top and bottom halves may have differences.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process the image
    processed_image, diff_mask = process_image(image)

    # Display the results
    st.image(processed_image, caption="Differences Highlighted", use_column_width=True)
    st.image(diff_mask, caption="Thresholded Differences", use_column_width=True)
