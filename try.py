import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuration & Page Setup ---
st.set_page_config(
    page_title="AI Image Segmenter",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Semantic Image Segmentation")
st.markdown("""
This app uses a DeepLabV3 model to segment objects in your images. 
Upload an image below to see the AI identify different regions (people, cars, animals, etc.).
""")

# --- 2. Model Loading (Cached) ---
@st.cache_resource
def load_model():
    """
    Loads the DeepLabV3 ResNet50 model pretrained on COCO/Pascal VOC.
    Cached so it doesn't reload on every interaction.
    """
    # Use 'DEFAULT' weights for the best available pre-trained weights
    model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
    model.eval()  # Set to evaluation mode
    return model

# Load the model immediately
with st.spinner("Loading AI Model... (this may take a moment first time)"):
    model = load_model()

# --- 3. Preprocessing Utils ---
def preprocess_image(image):
    """
    Prepares the uploaded image for the PyTorch model.
    """
    # Define standard transforms for pre-trained PyTorch models
    preprocess = transforms.Compose([
        transforms.Resize(520),              # Resize shortest side to 520
        transforms.ToTensor(),               # Convert to tensor (0-1)
        transforms.Normalize(                # Normalize with ImageNet mean/std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    input_tensor = preprocess(image)
    return input_tensor.unsqueeze(0) # Add batch dimension

# --- 4. Visualization Utils ---
def get_color_map():
    """
    Returns a dictionary of colors and class names for Pascal VOC dataset.
    """
    # Standard Pascal VOC classes (21 classes)
    VOC_CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "potted plant", "sheep",
        "sofa", "train", "tv/monitor"
    ]
    
    # Generate distinct colors for each class
    # We use a fixed seed so colors are consistent
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(21, 3))
    colors[0] = [0, 0, 0] # Background is black
    
    return VOC_CLASSES, colors

def decode_segmentation_masks(mask_tensor, colors):
    """
    Converts the raw model output (indices) into a colorful image.
    """
    # Move to CPU and convert to numpy
    r = Image.fromarray(mask_tensor.byte().cpu().numpy())
    
    # Create a flat palette list
    palette = []
    for color in colors:
        palette.extend(color)
        
    # Pad the palette to 256 colors (required by PIL putpalette)
    palette.extend([0, 0, 0] * (256 - len(colors)))
    
    r.putpalette(palette)
    return r.convert('RGB')

# --- 5. Main App Logic ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display original
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    # Run Inference
    with col2:
        st.subheader("2. Segmentation Result")
        
        # Add a "Segment" button so user can control when to run
        if st.button("Apply Segmentation", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Preprocess
                status_text.text("Preprocessing image...")
                progress_bar.progress(20)
                input_batch = preprocess_image(image)
                
                # Step 2: Inference
                status_text.text("Running Neural Network...")
                progress_bar.progress(50)
                
                if torch.cuda.is_available():
                    input_batch = input_batch.to('cuda')
                    model.to('cuda')

                with torch.no_grad():
                    output = model(input_batch)['out'][0]
                
                progress_bar.progress(80)
                status_text.text("Processing masks...")
                
                # Step 3: Post-process
                output_predictions = output.argmax(0)
                
                # Get colors and create mask image
                voc_classes, voc_colors = get_color_map()
                segmentation_mask = decode_segmentation_masks(output_predictions, voc_colors)
                
                # Step 4: Blend images (optional overlay)
                # Resize mask to match original image size exactly
                segmentation_mask = segmentation_mask.resize(image.size)
                blended = Image.blend(image, segmentation_mask, alpha=0.5)
                
                progress_bar.progress(100)
                status_text.empty() # Clear status
                
                # Display Result
                st.image(segmentation_mask, caption="Segmentation Mask", use_container_width=True)
                
                with st.expander("See Legend (Classes)"):
                    # Create a simple legend using markdown
                    legend_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
                    unique_classes = torch.unique(output_predictions).tolist()
                    
                    for cls_idx in unique_classes:
                        if cls_idx < len(voc_classes):
                            color = voc_colors[cls_idx]
                            name = voc_classes[cls_idx]
                            rgb_str = f"rgb({color[0]},{color[1]},{color[2]})"
                            legend_html += f"""
                                <div style="display:flex; align_items:center; margin-right:15px;">
                                    <div style="width:20px; height:20px; background-color:{rgb_str}; border-radius:50%; margin-right:5px; border:1px solid #ccc;"></div>
                                    <span>{name}</span>
                                </div>
                            """
                    legend_html += "</div>"
                    st.markdown(legend_html, unsafe_allow_html=True)
                    
                st.success("Segmentation Complete!")
                
                # Option to download
                st.download_button(
                    label="Download Mask",
                    data=segmentation_mask.tobytes(), # In real app, save to buffer first
                    file_name="segmentation_mask.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    with col2:
        st.info("Upload an image on the left to get started!")