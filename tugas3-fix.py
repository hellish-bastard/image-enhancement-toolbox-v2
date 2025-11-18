import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from streamlit_cropper import st_cropper

st.set_page_config(page_title="Image Enhancement Toolbox v2", layout="wide")

# Initialize session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'processed_image_clean' not in st.session_state:
    st.session_state.processed_image_clean = None
if 'roi_coords' not in st.session_state:
    st.session_state.roi_coords = None
if 'use_roi' not in st.session_state:
    st.session_state.use_roi = False
if 'cropped_area' not in st.session_state:
    st.session_state.cropped_area = None
if 'show_roi_box' not in st.session_state:
    st.session_state.show_roi_box = True
if 'reset_trigger' not in st.session_state:
    st.session_state.reset_trigger = 0

def apply_filters(image, use_gaussian, gaussian_kernel, use_median, median_kernel, 
                  use_sharpen, sharpen_strength):
    """Apply selected filters to the image"""
    result = image.copy()
    
    if use_gaussian:
        k = gaussian_kernel if gaussian_kernel % 2 == 1 else gaussian_kernel + 1
        result = cv2.GaussianBlur(result, (k, k), 0)
    
    if use_median:
        k = median_kernel if median_kernel % 2 == 1 else median_kernel + 1
        result = cv2.medianBlur(result, k)
    
    if use_sharpen:
        s = sharpen_strength
        kernel = np.array([[-1, -1, -1],
                           [-1, 8 + s, -1],
                           [-1, -1, -1]]) * (s / 2)
        result = cv2.filter2D(result, -1, kernel)
    
    return result

def apply_transformation(image, transform_type, histeq_strength):
    """Apply intensity transformation"""
    if transform_type == "None":
        return image
    
    elif transform_type == "Histogram Equalization":
        alpha = histeq_strength
        
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb_eq = ycrcb.copy()
        ycrcb_eq[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        eq_img = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)
        
        blended = cv2.addWeighted(eq_img, alpha, image, 1 - alpha, 0)
        return blended
    
    return image

def apply_morphology(image, morph_type, morph_kernel):
    """Apply morphological operations"""
    if morph_type == "None":
        return image
    
    k = morph_kernel if morph_kernel % 2 == 1 else morph_kernel + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    
    if morph_type == "Opening":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif morph_type == "Closing":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif morph_type == "Top-Hat":
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    
    return image

def process_image_with_roi(original_image, roi_coords, use_roi, 
                           use_gaussian, gaussian_kernel, use_median, median_kernel,
                           use_sharpen, sharpen_strength, transform_type, histeq_strength,
                           morph_type, morph_kernel):
    """Process image with ROI support"""
    result = original_image.copy()
    
    if use_roi and roi_coords is not None:
        x1, y1, x2, y2 = roi_coords
        
        # Validate coordinates
        h, w = result.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # Extract ROI
        roi = result[y1:y2, x1:x2].copy()
        
        # Apply processing only to ROI
        roi = apply_filters(roi, use_gaussian, gaussian_kernel, use_median, 
                           median_kernel, use_sharpen, sharpen_strength)
        roi = apply_transformation(roi, transform_type, histeq_strength)
        roi = apply_morphology(roi, morph_type, morph_kernel)
        
        # Put processed ROI back
        result[y1:y2, x1:x2] = roi
        
        
    else:
        # Apply processing to entire image
        result = apply_filters(result, use_gaussian, gaussian_kernel, use_median, 
                              median_kernel, use_sharpen, sharpen_strength)
        result = apply_transformation(result, transform_type, histeq_strength)
        result = apply_morphology(result, morph_type, morph_kernel)
    
    return result

def plot_histogram(image, title="Histogram"):
    """Create histogram plot for RGB image"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = ('red', 'green', 'blue')
    channel_names = ('Red', 'Green', 'Blue')
    
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, label=name, alpha=0.7, linewidth=2)
    
    ax.set_xlim([0, 256])
    ax.set_xlabel('Pixel Intensity', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Main App
st.title("Image Enhancement Toolbox v2")
st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Upload Image
    uploaded_file = st.file_uploader("📁 Upload Image", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
    
    if uploaded_file is not None:
        # Read and convert image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.session_state.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Reset ROI when new image uploaded
        st.session_state.roi_coords = None
    
    st.markdown("---")
    
    # Only show controls if image is loaded
    if st.session_state.original_image is not None:
        # ROI SELECTION
        st.subheader("🎯 Region of Interest (ROI)")
        
        # Determine default value for checkbox based on reset trigger
        default_use_roi = False if st.session_state.reset_trigger > 0 else st.session_state.use_roi
        
        use_roi = st.checkbox("Enable ROI (Edit Selected Area Only)", 
                             value=default_use_roi, 
                             key=f"use_roi_checkbox_{st.session_state.reset_trigger}")
        st.session_state.use_roi = use_roi
        
        if use_roi:
            st.info("👇 Drag the box, then click 'Lock ROI' button")
            
            if st.session_state.roi_coords is not None:
                x1, y1, x2, y2 = st.session_state.roi_coords
                st.success(f"✅ ROI Locked: ({x1}, {y1}) → ({x2}, {y2})")
                st.info(f"📏 Size: {x2-x1} × {y2-y1} pixels")
            else:
                st.warning("⚠️ ROI not locked yet. Drag box below, then click 'Lock ROI'")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔒 Lock ROI", use_container_width=True, type="primary"):
                    # ROI will be locked from cropper output
                    st.success("ROI Locked!")
            with col_btn2:
                if st.button("❌ Reset ROI", use_container_width=True):
                    st.session_state.roi_coords = None
                    st.session_state.cropped_area = None
                    st.rerun()
        else:
            st.session_state.roi_coords = None
        
        st.markdown("---")
        
        # 1. FILTERING
        st.subheader("1️⃣ Filtering")
        
        use_gaussian = st.checkbox("Gaussian Blur", value=False, 
                                   key=f"gaussian_check_{st.session_state.reset_trigger}")
        if use_gaussian:
            gaussian_kernel = st.slider("Gaussian Kernel Size", 
                                       min_value=3, max_value=15, value=5, step=2, 
                                       key=f"gaussian_slider_{st.session_state.reset_trigger}")
        else:
            gaussian_kernel = 5
        
        use_median = st.checkbox("Median Filter", value=False, 
                                key=f"median_check_{st.session_state.reset_trigger}")
        if use_median:
            median_kernel = st.slider("Median Kernel Size", 
                                     min_value=3, max_value=15, value=5, step=2, 
                                     key=f"median_slider_{st.session_state.reset_trigger}")
        else:
            median_kernel = 5
        
        use_sharpen = st.checkbox("Sharpening", value=False, 
                                 key=f"sharpen_check_{st.session_state.reset_trigger}")
        if use_sharpen:
            sharpen_strength = st.slider("Sharpen Strength", 
                                        min_value=0.5, max_value=3.0, value=1.0, step=0.1, 
                                        key=f"sharpen_slider_{st.session_state.reset_trigger}")
        else:
            sharpen_strength = 1.0
        
        st.markdown("---")
        
        # 2. INTENSITY TRANSFORM
        st.subheader("2️⃣ Intensity Transform")
        transform_type = st.radio("Select Transform:", 
                                  ["None", "Histogram Equalization"], 
                                  key=f"transform_radio_{st.session_state.reset_trigger}")
        
        if transform_type == "Histogram Equalization":
            histeq_strength = st.slider("Equalization Level", 
                                       min_value=0.0, max_value=1.0, value=1.0, step=0.01, 
                                       key=f"histeq_slider_{st.session_state.reset_trigger}")
        else:
            histeq_strength = 1.0
        
        st.markdown("---")
        
        # 3. MORPHOLOGICAL OPERATIONS
        st.subheader("3️⃣ Morphological Operations")
        morph_type = st.radio("Select Operation:", 
                             ["None", "Opening", "Closing", "Top-Hat"], 
                             key=f"morph_radio_{st.session_state.reset_trigger}")
        
        if morph_type != "None":
            morph_kernel = st.slider("Morphology Kernel Size", 
                                    min_value=3, max_value=15, value=5, step=2, 
                                    key=f"morph_slider_{st.session_state.reset_trigger}")
        else:
            morph_kernel = 5
        
        st.markdown("---")
        
        # Reset Button
        if st.button("🔄 Reset All", use_container_width=True):
            # Reset ROI
            st.session_state.roi_coords = None
            st.session_state.use_roi = False
            st.session_state.cropped_area = None
            if hasattr(st.session_state, 'temp_roi'):
                del st.session_state.temp_roi
            
            # Reset processed image
            st.session_state.processed_image = None
            
            # Increment reset trigger to force re-creation of all widgets
            st.session_state.reset_trigger += 1
            
            st.rerun()

# Main content area
if st.session_state.original_image is None:
    st.info("👆 Please upload an image to get started!")
else:
    # ROI Selection with Drag
    if st.session_state.use_roi:
        st.subheader("🎯 Step 1: Drag to Select ROI")
        st.info("👆 Drag the corners or sides of the red box to adjust the area, then click 'Lock ROI' in sidebar")
        
        # Convert to PIL Image
        pil_image = Image.fromarray(st.session_state.original_image)
        
        # Use cropper to select ROI - store in temp variable
        cropped_result = st_cropper(
            pil_image,
            realtime_update=True,
            box_color='#FF0000',
            aspect_ratio=None,
            return_type='box',
            key=f'roi_cropper_{st.session_state.reset_trigger}'
        )
        
        # Store temp coordinates but don't lock yet
        if cropped_result is not None and isinstance(cropped_result, dict):
            left = cropped_result.get('left', 0)
            top = cropped_result.get('top', 0)
            width = cropped_result.get('width', 100)
            height = cropped_result.get('height', 100)
            
            # Calculate coordinates
            temp_x1 = int(left)
            temp_y1 = int(top)
            temp_x2 = int(left + width)
            temp_y2 = int(top + height)
            
            # Store as temp (will be locked by button)
            st.session_state.temp_roi = (temp_x1, temp_y1, temp_x2, temp_y2)
        
        st.markdown("---")
        
        st.subheader("🎯 Step 2: Apply Filters")
        if st.session_state.roi_coords is not None:
            st.success("✅ ROI is locked!")
        else:
            st.warning("⚠️ Please lock ROI first")
    
    # Auto-lock ROI when Lock button is clicked (from sidebar)
    if st.session_state.use_roi and hasattr(st.session_state, 'temp_roi'):
        # Check if we should lock the ROI
        if st.session_state.roi_coords is None:
            # Auto-lock if temp_roi exists
            st.session_state.roi_coords = st.session_state.temp_roi
    
    # Process image
    if st.session_state.original_image is not None:
        st.session_state.processed_image = process_image_with_roi(
            st.session_state.original_image,
            st.session_state.roi_coords,
            st.session_state.use_roi,
            use_gaussian, gaussian_kernel,
            use_median, median_kernel,
            use_sharpen, sharpen_strength,
            transform_type, histeq_strength,
            morph_type, morph_kernel
        )
    
    # Display Images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(st.session_state.original_image, use_container_width=True)
    
    with col2:
        st.subheader("✨ Processed Image")
        if st.session_state.processed_image is not None:
            st.image(st.session_state.processed_image, use_container_width=True)
            
            if st.session_state.use_roi and st.session_state.roi_coords is not None:
                st.info("🟢 Green box shows processed ROI area")
            
            # Download button
            result_bgr = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.png', result_bgr)
            
            st.download_button(
                label="💾 Download Result",
                data=buffer.tobytes(),
                file_name="enhanced_image.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.info("No processing applied yet")
    
    # Display Histograms
    st.markdown("---")
    st.subheader("📊 Histogram Analysis (RGB)")
    
    hist_col1, hist_col2 = st.columns(2)
    
    with hist_col1:
        st.markdown("**📈 Histogram Sebelum (Original)**")
        fig_original = plot_histogram(st.session_state.original_image, "Original Image Histogram")
        st.pyplot(fig_original)
        plt.close(fig_original)
    
    with hist_col2:
        st.markdown("**📈 Histogram Sesudah (Processed)**")
        if st.session_state.processed_image is not None:
            fig_processed = plot_histogram(st.session_state.processed_image, "Processed Image Histogram")
            st.pyplot(fig_processed)
            plt.close(fig_processed)
        else:
            st.info("No processing applied yet")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Image Enhancement Toolbox V2"
    "</div>",
    unsafe_allow_html=True
)