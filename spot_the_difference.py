import streamlit as st
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import correlate

# Helper Functions

def align_images(image1, image2):
    """Align two images using cross-correlation to minimize pixel difference."""
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Perform cross-correlation
    result = correlate(gray1, gray2, mode='constant')
    y, x = np.unravel_index(np.argmax(result), result.shape)

    # Create translation matrix and apply
    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
    aligned_image = cv2.warpAffine(image2, translation_matrix, (image1.shape[1], image1.shape[0]))

    return aligned_image

def find_differences(image1, image2, threshold, color, thickness, max_differences=None):
    """Find and mark differences between two images."""
    diff = cv2.absdiff(image1, image2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_image = image1.copy()
    differences_found = 0

    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Ignore very small differences
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            radius = max(w, h) // 2
            cv2.circle(result_image, center, radius, color, thickness)
            differences_found += 1

            # Stop if we've found the desired number of differences
            if max_differences and differences_found >= max_differences:
                break

    return result_image, diff, differences_found

# Streamlit App

st.title("Spot the Difference")

# Upload Section
st.subheader("Step 1: Upload Images")
option = st.radio(
    "Choose your image input method",
    ["Split one image (Top/Bottom or Left/Right)", "Upload two separate images"],
    help="Choose how to provide input images."
)

if option == "Split one image (Top/Bottom or Left/Right)":
    uploaded_file = st.file_uploader("Upload an image to split", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        height, width, _ = image_np.shape
        if height > width:
            # Top/Bottom split
            half_height = height // 2
            image1 = image_np[:half_height, :]
            image2 = image_np[half_height:, :]
        else:
            # Left/Right split
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
    st.sidebar.subheader("Step 2: Choose Difference Detection Settings")

    # Number of differences to find
    max_differences = st.sidebar.number_input(
        "Number of differences to find", min_value=0, value=0, help="Enter the number of differences to detect. 0 for all."
    )

    st.sidebar.subheader("Step 3: Adjust Detection Parameters")
    threshold = st.sidebar.slider("Threshold for differences", 1, 255, 50, help="Lower values detect more subtle differences.")
    color = st.sidebar.color_picker("Marking color", "#FF0000", help="Choose the color to highlight differences.")
    thickness = st.sidebar.slider("Marking thickness", 1, 10, 3, help="Adjust the thickness of the markers on differences.")

    st.sidebar.subheader("Step 4: Alignment Method")
    st.sidebar.write(
        "Images will be automatically aligned for optimal comparison. You can adjust alignment settings here if needed."
    )
    
    # Align Images
    aligned_image2 = align_images(image1, image2)
    
    # Find differences
    result_image, diff_image, differences_found = find_differences(
        image1, aligned_image2, threshold, tuple(int(color[i:i+2], 16) for i in (1, 3, 5)), thickness, max_differences or None
    )

    # Display Results
    st.subheader("Results")
    st.write(f"Total differences found: {differences_found}")
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

    # Show Process Explanation
    st.sidebar.subheader("How it works:")
    st.sidebar.write(
        """
        - **Step 1:** Upload images (either split one or upload two separate).
        - **Step 2:** Choose how many differences you'd like to find (0 for all).
        - **Step 3:** Adjust detection threshold, marking color, and thickness for better visuals.
        - **Step 4:** Alignment is performed automatically. You can adjust settings here.
        """
    )
