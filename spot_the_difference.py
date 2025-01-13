import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Helper Functions

def align_images_by_template_matching(image1, image2):
    """Align two images using template matching (cross-correlation)."""
    # Convert images to grayscale for better processing
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Perform template matching (cross-correlation)
    result = cv2.matchTemplate(gray2, gray1, cv2.TM_CCOEFF_NORMED)

    # Get the location with the maximum correlation (best alignment)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Create the affine matrix for shifting
    translation_matrix = np.float32([[1, 0, max_loc[0]], [0, 1, max_loc[1]]])

    # Apply the translation using warpAffine
    aligned_image2 = cv2.warpAffine(image2, translation_matrix, (image2.shape[1], image2.shape[0]))

    return aligned_image2, max_loc, max_val

def find_differences_by_pixel(image1, image2, threshold, color, thickness, max_differences=None):
    """Find and mark differences between two images by pixel difference."""
    # Compute absolute difference
    diff = cv2.absdiff(image1, image2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Threshold the difference to get the regions with significant changes
    _, thresh = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded difference
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

            # Stop if the desired number of differences is found
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

    # Align Images using Template Matching (Cross-Correlation)
    aligned_image2, max_loc, max_val = align_images_by_template_matching(image1, image2)
    
    # Find differences by pixel
    result_image, diff_image, differences_found = find_differences_by_pixel(
        image1, aligned_image2, threshold, tuple(int(color[i:i+2], 16) for i in (1, 3, 5)), thickness, max_differences or None
    )

    # Display Results
    st.subheader("Results")
    st.write(f"Total differences found: {differences_found}")
    st.write(f"Best alignment position: {max_loc} with match value: {max_val:.4f}")
    
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
        **Step 1:** Upload or split your images.
        - You can upload two separate images or split one image into two (top/bottom or left/right).

        **Step 2:** Choose the number of differences you want to detect.
        - Set the number of differences (enter 0 for detecting all).

        **Step 3:** Adjust the detection parameters.
        - Threshold: Controls the sensitivity of difference detection.
        - Color: Choose the color to mark the differences.
        - Thickness: Adjust the thickness of the circle markers for differences.

        **Step 4:** Image Alignment (Template Matching).
        - The second image is aligned to the first using template matching (cross-correlation).

        **Step 5:** Difference Detection.
        - The pixel-by-pixel differences are detected and marked.

        **Step 6:** Results.
        - View the marked differences and download the highlighted image and difference mask.

        Enjoy spotting the differences!
        """
    )
