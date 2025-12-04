import streamlit as st
from PIL import Image, ImageOps

st.title("bw_filter_app")

# 1. Upload the image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    
    # Show the original
    st.write("### Original Image")
    st.image(image, use_container_width=True)

    # 2. Apply ONE filter (Grayscale)
    # This converts the image to Black & White
    filtered_image = ImageOps.grayscale(image)

    # Show the result
    st.write("### Black & White Filter")
    st.image(filtered_image, use_container_width=True)

    # 3. Download button
    # Save to a buffer to allow download
    from io import BytesIO
    buf = BytesIO()
    filtered_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download B&W Image",
        data=byte_im,
        file_name="bw_image.png",
        mime="image/png"
    )
