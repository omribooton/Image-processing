import streamlit as st
import numpy as np
from PIL import Image
from scikit-image import filters, color, util

st.title("skimage_filter_app")

# 1. Upload the image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image with PIL
    image = Image.open(uploaded_file)
    
    # Show the original
    st.write("### Original Image")
    st.image(image, use_container_width=True)

    # 2. Apply ONE filter using scikit-image (Sobel Edge Detection)
    # Convert PIL image to Numpy Array
    image_array = np.array(image)

    # Convert to grayscale first (Sobel works on 2D arrays)
    # Note: If image is already gray, rgb2gray handles it or we check dimensions
    if len(image_array.shape) == 3:
        gray_image = color.rgb2gray(image_array)
    else:
        gray_image = image_array

    # Apply Sobel Filter (Edge Detection)
    filtered_array = filters.sobel(gray_image)

    # Convert back to 8-bit integer for display (0-255)
    filtered_image_uint8 = util.img_as_ubyte(filtered_array)
    
    # Convert back to PIL image
    final_image = Image.fromarray(filtered_image_uint8)

    # Show the result
    st.write("### Edge Detection (Sobel) Filter")
    st.image(final_image, use_container_width=True)

    # 3. Download button
    from io import BytesIO
    buf = BytesIO()
    final_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Filtered Image",
        data=byte_im,
        file_name="edge_detected_image.png",
        mime="image/png"
    )

