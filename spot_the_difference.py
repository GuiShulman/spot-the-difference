import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Helper Functions
def crop_borders(image):
    """Crop unnecessary borders (black or white) from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y + h, x:x + w]

def align_images(image1, image2, move_range=50):
    """Align two images by maximizing pixel similarity within a movement range."""
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    best_score = -1
    best_dx, best_dy = 0, 0
    
    for dx in range(-move_range, move_range + 1):
        for dy in range(-move_range, move_range + 1):
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(gray2, matrix, (gray2.shape[1], gray2.shape[0]))
            score = np.sum(gray1 == shifted)
            
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

    matrix = np.float32([[1, 0, best_dx], [0, 1, best_dy]])
    aligned = cv2.warpAffine(image2, matrix, (image2.shape[1], image2.shape[0]))
    return aligned, best_dx, best_dy

def find_differences(image1, image2, threshold=50, min_area=50, color=(0, 0, 255), thickness=2):
    """Find differences between two images and highlight them."""
    diff = cv2.absdiff(image1, image2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = image1.copy()
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            radius = max(w, h) // 2
            cv2.circle(result, center, radius, color, thickness)
    
    return result, diff

# Streamlit App
st.title("Spot the Difference")

# Upload Images
st.subheader("Upload Images")
uploaded_files = st.file_uploader("Upload two images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 2:
    image1 = np.array(Image.open(uploaded_files[0]))
    image2 = np.array(Image.open(uploaded_files[1]))
    
    st.image([image1, image2], caption=["Image 1", "Image 2"], use_container_width=True)

    # Settings
    st.sidebar.title("Settings")
    crop = st.sidebar.checkbox("Crop Borders", value=True, help="Remove unnecessary black/white borders.")
    move_range = st.sidebar.slider("Alignment Range", 1, 100, 50, help="Max pixel range to align images.")
    threshold = st.sidebar.slider("Difference Sensitivity", 1, 255, 50, help="Sensitivity of difference detection.")
    min_area = st.sidebar.slider("Minimum Difference Area", 1, 500, 50, help="Ignore small differences.")
    show_diff_mask = st.sidebar.checkbox("Show Difference Mask", value=True, help="Display raw difference mask.")
    color = tuple(int(st.sidebar.color_picker("Highlight Color", "#FF0000")[i:i+2], 16) for i in (1, 3, 5))
    thickness = st.sidebar.slider("Circle Thickness", 1, 10, 3, help="Thickness of highlighted circles.")
    
    # Crop Borders
    if crop:
        image1 = crop_borders(image1)
        image2 = crop_borders(image2)
    
    # Align Images
    aligned_image2, dx, dy = align_images(image1, image2, move_range)
    st.sidebar.write(f"Alignment Offset: dx={dx}, dy={dy}")

    # Find Differences
    result_image, diff_image = find_differences(image1, aligned_image2, threshold, min_area, color, thickness)

    # Display Results
    st.subheader("Results")
    st.image(result_image, caption="Differences Highlighted", use_container_width=True)
    if show_diff_mask:
        st.image(diff_image, caption="Difference Mask", use_container_width=True)
