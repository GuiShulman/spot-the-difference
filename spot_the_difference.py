import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Helper function for aligning images by moving and scaling
def align_and_scale_images(image1, image2, align_type='None'):
    """Align the second image to the first one, scaling and shifting it if necessary."""
    # Convert both images to grayscale for better template matching
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Perform template matching (cross-correlation)
    result = cv2.matchTemplate(gray2, gray1, cv2.TM_CCOEFF_NORMED)

    # Get the location with the maximum correlation (best alignment)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # We assume a match value threshold for good alignment, e.g., 0.7
    if max_val < 0.7:  # Adjust threshold if needed
        raise ValueError("The match value is too low, alignment failed!")

    # Create the translation matrix for the shift
    translation_matrix = np.float32([[1, 0, max_loc[0]], [0, 1, max_loc[1]]])

    # Apply the translation using warpAffine
    aligned_image = cv2.warpAffine(image2, translation_matrix, (image1.shape[1], image1.shape[0]))

    # If user specified Top-Bottom or Left-Right shifting
    if align_type == 'Top-Bottom':
        aligned_image = cv2.copyMakeBorder(aligned_image, max_loc[1], image1.shape[0] - max_loc[1] - image2.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    elif align_type == 'Left-Right':
        aligned_image = cv2.copyMakeBorder(aligned_image, 0, 0, max_loc[0], image1.shape[1] - max_loc[0] - image2.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Scaling the image to fit if necessary (e.g., scaling the second image to match the first one)
    if image2.shape != image1.shape:
        scale_factor = min(image1.shape[0] / image2.shape[0], image1.shape[1] / image2.shape[1])
        new_size = (int(image2.shape[1] * scale_factor), int(image2.shape[0] * scale_factor))
        image2_resized = cv2.resize(aligned_image, new_size)
        aligned_image = cv2.resize(image2_resized, (image1.shape[1], image1.shape[0]))

    return aligned_image, max_loc, max_val

# Streamlit App
st.title("Image Alignment with Moving and Scaling")

# Upload Section
uploaded_files = st.file_uploader("Upload two images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 2:
    image1 = np.array(Image.open(uploaded_files[0]))
    image2 = np.array(Image.open(uploaded_files[1]))

    # Display uploaded images
    st.image([image1, image2], caption=["Image 1", "Image 2"], use_container_width=True)

    # Option for user to specify how to shift the second image
    align_type = st.selectbox("Select alignment option:", ["None", "Top-Bottom", "Left-Right"])

    # Align the images
    try:
        aligned_image, max_loc, max_val = align_and_scale_images(image1, image2, align_type)
        st.sidebar.write(f"Best alignment position: {max_loc} with match value: {max_val:.4f}")
        
        # Show aligned image
        st.subheader("Aligned Image 2")
        st.image(aligned_image, caption="Aligned Image 2", use_container_width=True)
        
        # Allow users to download the aligned image
        aligned_image_pil = Image.fromarray(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
        st.download_button(
            label="Download Aligned Image",
            data=aligned_image_pil.tobytes(),
            file_name="aligned_image.png",
            mime="image/png"
        )
        
    except ValueError as e:
        st.error(f"Error: {e}")
