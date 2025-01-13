import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Helper Functions

def align_images_by_pixel_similarity(image1, image2, move_range):
    """Align two images by maximizing pixel similarity within a defined movement range."""
    # Convert images to grayscale for comparison
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    best_score = -1
    best_dx, best_dy = 0, 0

    # Search for the best alignment within the user-defined range
    for dx in range(-move_range, move_range + 1):
        for dy in range(-move_range, move_range + 1):
            # Translate the second image
            translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted_image = cv2.warpAffine(gray2, translation_matrix, (gray2.shape[1], gray2.shape[0]))

            # Compute similarity score (count of identical pixels)
            score = np.sum(gray1 == shifted_image)

            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

    # Apply the best alignment
    translation_matrix = np.float32([[1, 0, best_dx], [0, 1, best_dy]])
    aligned_image2 = cv2.warpAffine(image2, translation_matrix, (image2.shape[1], image2.shape[0]))

    return aligned_image2, best_dx, best_dy, best_score

def find_differences_by_pixel(image1, image2, threshold, color, thickness, max_differences=None):
    """Find and mark differences between two images by pixel difference."""
    diff = cv2.absdiff(image1, image2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_image = image1.copy()
    differences_found = 0

    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Ignore small differences
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            radius = max(w, h) // 2
            cv2.circle(result_image, center, radius, color, thickness)
            differences_found += 1

            if max_differences and differences_found >= max_differences:
                break

    return result_image, diff, differences_found

# Streamlit App

st.title("Spot the Difference")

# Upload Section
st.subheader("Upload Images")
option = st.radio(
    "How would you like to provide input images?",
    ["Split one image (Top/Bottom or Left/Right)", "Upload two separate images"],
    help="Split a single image or upload two separate images for comparison."
)

if option == "Split one image (Top/Bottom or Left/Right)":
    uploaded_file = st.file_uploader("Upload an image to split", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        height, width, _ = image_np.shape
        if height > width:
            half_height = height // 2
            image1 = image_np[:half_height, :]
            image2 = image_np[half_height:, :]
        else:
            half_width = width // 2
            image1 = image_np[:, :half_width]
            image2 = image_np[:, half_width:]

        st.image([image1, image2], caption=["Image 1", "Image 2"], use_container_width=True)

elif option == "Upload two separate images":
    uploaded_files = st.file_uploader(
        "Upload exactly two images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    if uploaded_files and len(uploaded_files) == 2:
        image1 = np.array(Image.open(uploaded_files[0]))
        image2 = np.array(Image.open(uploaded_files[1]))
        st.image([image1, image2], caption=["Image 1", "Image 2"], use_container_width=True)

# Settings Section
if "image1" in locals() and "image2" in locals():
    st.sidebar.title("Settings")

    # Alignment Parameters
    st.sidebar.subheader("Image Alignment")
    move_range = st.sidebar.slider(
        "Max Pixel Movement Range", 1, 100, 50,
        help="Defines the maximum range (in pixels) to search for alignment between images."
    )

    # Difference Detection Parameters
    st.sidebar.subheader("Difference Detection")
    threshold = st.sidebar.slider(
        "Detection Sensitivity", 1, 255, 50,
        help="Lower values detect more subtle differences, higher values detect only significant changes."
    )
    color = st.sidebar.color_picker(
        "Highlight Color", "#FF0000",
        help="Choose the color used to highlight the differences."
    )
    thickness = st.sidebar.slider(
        "Marker Thickness", 1, 10, 3,
        help="Adjust the thickness of the circle markers used to highlight differences."
    )
    max_differences = st.sidebar.number_input(
        "Max Differences to Highlight", min_value=0, value=0,
        help="Set the maximum number of differences to highlight. Enter 0 to highlight all differences."
    )

    # Alignment Process
    aligned_image2, best_dx, best_dy, best_score = align_images_by_pixel_similarity(image1, image2, move_range)
    st.sidebar.write(f"Alignment Offset: dx={best_dx}, dy={best_dy}")
    st.sidebar.write(f"Pixel Similarity Score: {best_score}")

    # Difference Detection Process
    result_image, diff_image, differences_found = find_differences_by_pixel(
        image1, aligned_image2, threshold,
        tuple(int(color[i:i+2], 16) for i in (1, 3, 5)),
        thickness, max_differences or None
    )

    # Display Results
    st.subheader("Results")
    st.write(f"Total Differences Found: {differences_found}")

    col1, col2 = st.columns(2)
    with col1:
        st.image(result_image, caption="Differences Highlighted", use_container_width=True)
        st.download_button(
            label="Download Highlighted Image",
            data=Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)).tobytes(),
            file_name="highlighted_differences.png",
            mime="image/png"
        )
    with col2:
        st.image(diff_image, caption="Difference Mask", use_container_width=True)
        st.download_button(
            label="Download Difference Mask",
            data=Image.fromarray(diff_image).tobytes(),
            file_name="difference_mask.png",
            mime="image/png"
        )
