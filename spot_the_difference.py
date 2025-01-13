import streamlit as st
import cv2
import numpy as np
from PIL import Image


def align_images(image1, image2):
    """Align two images using keypoint matching."""
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    height, width, _ = image1.shape
    aligned_image = cv2.warpPerspective(image2, matrix, (width, height))
    return aligned_image


def find_differences(image1, image2, threshold, min_size, mark_type):
    """Find and mark differences between two images."""
    diff = cv2.absdiff(image1, image2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_image = image1.copy()
    for contour in contours:
        if cv2.contourArea(contour) > min_size:
            x, y, w, h = cv2.boundingRect(contour)
            if mark_type == "Rectangle":
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            elif mark_type == "Circle":
                center = (x + w // 2, y + h // 2)
                radius = max(w, h) // 2
                cv2.circle(result_image, center, radius, (0, 255, 0), 2)

    return result_image, diff


# Streamlit App
st.title("Spot the Difference - User-Friendly")

# Step 1: Image Upload
st.subheader("Step 1: Upload Images")
option = st.radio(
    "Choose input method:",
    ["Split one image", "Upload two images"],
    help="Split one image into two parts or upload two separate images."
)

if option == "Split one image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    split_direction = st.radio(
        "How to split the image?",
        ["Top/Bottom", "Left/Right"],
        help="Choose whether to split the image vertically or horizontally."
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        height, width, _ = image_np.shape
        if split_direction == "Top/Bottom":
            half_height = height // 2
            image1 = image_np[:half_height, :]
            image2 = image_np[half_height:, :]
        else:
            half_width = width // 2
            image1 = image_np[:, :half_width]
            image2 = image_np[:, half_width:]

        st.image([image1, image2], caption=["Image 1", "Image 2"], use_container_width=True)

elif option == "Upload two images":
    uploaded_file1 = st.file_uploader("Upload the first image", type=["jpg", "jpeg", "png"], key="image1")
    uploaded_file2 = st.file_uploader("Upload the second image", type=["jpg", "jpeg", "png"], key="image2")

    if uploaded_file1 and uploaded_file2:
        image1 = np.array(Image.open(uploaded_file1))
        image2 = np.array(Image.open(uploaded_file2))

        st.image([image1, image2], caption=["Image 1", "Image 2"], use_container_width=True)

# Step 2: Parameters
if "image1" in locals() and "image2" in locals():
    st.subheader("Step 2: Adjust Parameters")
    threshold = st.slider("Threshold for differences:", 1, 255, 50, help="Lower values detect smaller differences.")
    min_size = st.slider("Minimum size of differences (pixels):", 1, 500, 50, help="Ignore small differences.")
    mark_type = st.radio("Mark differences as:", ["Rectangle", "Circle"], help="Choose how to highlight differences.")

    # Step 3: Process Images
    st.subheader("Step 3: View Results")
    aligned_image2 = align_images(image1, image2)
    result_image, diff_image = find_differences(image1, aligned_image2, threshold, min_size, mark_type)

    st.image(result_image, caption="Differences Highlighted", use_container_width=True)
    st.image(diff_image, caption="Difference Mask", use_container_width=True)

    # Step 4: Download Results
    st.subheader("Step 4: Download Results")
    result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    diff_pil = Image.fromarray(diff_image)

    st.download_button(
        label="Download Highlighted Image",
        data=result_pil.tobytes(),
        file_name="highlighted_differences.png",
        mime="image/png"
    )
    st.download_button(
        label="Download Difference Mask",
        data=diff_pil.tobytes(),
        file_name="difference_mask.png",
        mime="image/png"
    )
