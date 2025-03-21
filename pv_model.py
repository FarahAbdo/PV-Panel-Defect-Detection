import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

# Set page configuration
st.set_page_config(
    page_title="PV Panel Defect Detection",
    page_icon="🔍",
    layout="wide"
)

# Define the defect classes (limited to the ones in the dataset)
CLASS_NAMES = [
    "Examined", 
    "ShortCircuitString", 
    "ShortCircuitCell (LowPowerCell)", 
    "Crack", 
    "MicroCrack", 
    "OtherError"
]

# Color mapping for different defect types (BGR format for OpenCV)
COLORS = [
    (0, 255, 0),     # Green for Examined
    (0, 0, 255),     # Red for ShortCircuitString
    (255, 0, 0),     # Blue for ShortCircuitCell
    (0, 255, 255),   # Yellow for Crack
    (255, 0, 255),   # Magenta for MicroCrack
    (255, 255, 0)    # Cyan for OtherError
]

# Function to load the model
@st.cache_resource
def load_model(model_path):
    """Load and return the YOLO model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Function to perform inference on an image
def detect_defects(model, image, conf_threshold=0.25):
    """
    Detect defects in the given image using the model
    Args:
        model: Loaded YOLO model
        image: Image to detect defects in (numpy array in BGR format)
        conf_threshold: Confidence threshold for detections
    Returns:
        Annotated image, results object, and defect summary
    """
    # Make a copy of the image for drawing on
    annotated_img = image.copy()
    
    # Run inference
    try:
        # Convert BGR to RGB for YOLO model
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(rgb_image, conf=conf_threshold, verbose=False)[0]  # Get the first result
        
        # Create a defect summary
        defect_counts = {class_name: 0 for class_name in CLASS_NAMES}
        
        # Check if we have any detections
        if results is not None and results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get class ID and confidence
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                
                # Ensure class ID is within range
                if 0 <= cls_id < len(CLASS_NAMES):
                    # Get class name and color
                    class_name = CLASS_NAMES[cls_id]
                    color = COLORS[cls_id]
                    
                    # Draw rectangle
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{class_name}: {conf:.2f}"
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    y1 = max(y1, label_height + 10)
                    cv2.rectangle(annotated_img, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
                    cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Update defect count
                    defect_counts[class_name] += 1
                else:
                    st.warning(f"Detected class ID {cls_id} is out of range. Skipping this detection.")
        
        return annotated_img, results, defect_counts
    
    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
        return annotated_img, None, {class_name: 0 for class_name in CLASS_NAMES}

# Function to create a defect summary
def create_defect_summary(defect_counts):
    """Create a summary of detected defects"""
    st.subheader("Defect Summary")
    
    # Filter out classes with zero counts
    found_defects = {cls: count for cls, count in defect_counts.items() if count > 0}
    
    if not found_defects:
        st.warning("No defects detected in the image. Try adjusting the confidence threshold or try a different image.")
        return
    
    # Display defect counts
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Defect Counts:**")
        for cls, count in found_defects.items():
            st.write(f"- {cls}: {count}")
    
    with col2:
        # Create a simple bar chart for detected defects
        if found_defects:
            fig, ax = plt.subplots(figsize=(10, 5))
            classes = list(found_defects.keys())
            counts = list(found_defects.values())
            
            # Safely get colors for each class
            colors = []
            for cls in classes:
                try:
                    idx = CLASS_NAMES.index(cls)
                    colors.append(rgba_to_hex(COLORS[idx]))
                except ValueError:
                    colors.append('#777777')  # Default gray if class not found
            
            bars = ax.bar(classes, counts, color=colors)
            ax.set_ylabel('Count')
            ax.set_title('Defect Counts')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

# Helper function to convert RGB to hex color
def rgba_to_hex(rgba):
    """Convert RGB or RGBA to hex color string"""
    b, g, r = rgba[:3]
    return f'#{r:02x}{g:02x}{b:02x}'

# Function to display model information and debug data
def display_debug_info(model, results):
    with st.expander("Debug Information"):
        st.write("**Model Information:**")
        st.write(f"Model Type: {type(model).__name__}")
        st.write(f"Model Task: {getattr(model, 'task', 'Unknown')}")
        
        if results is not None:
            st.write("\n**Detection Results:**")
            st.write(f"Result Type: {type(results).__name__}")
            if hasattr(results, 'boxes') and results.boxes is not None:
                st.write(f"Number of detections: {len(results.boxes)}")
                if len(results.boxes) > 0:
                    st.write("\nFirst few detections:")
                    for i, box in enumerate(results.boxes[:3]):  # Show first 3 detections
                        cls_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        class_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"Unknown ({cls_id})"
                        st.write(f"Detection {i+1}: Class={class_name}, Confidence={conf:.3f}")
            else:
                st.write("No detection boxes found in results.")
        else:
            st.write("No results available. Model inference may have failed.")

# Generate a unique filename for the temporary file
def get_unique_filename(suffix='.png'):
    """Generate a unique filename using timestamp and random number"""
    timestamp = int(time.time() * 1000)
    random_num = np.random.randint(1000, 9999)
    return f"pv_defect_{timestamp}_{random_num}{suffix}"

# Main function to create the Streamlit app
def main():
    # App title and description
    st.title("PV Panel Defect Detection")
    st.markdown("""
    Upload an image of a photovoltaic panel to detect possible defects.
    This application uses a YOLOv8 model trained to identify 6 different types of PV panel defects.
    """)
    
    # Sidebar for defect classes information only
    st.sidebar.header("Defect Classes")
    for i, class_name in enumerate(CLASS_NAMES):
        color_hex = rgba_to_hex(COLORS[i])
        st.sidebar.markdown(f'<div style="display: flex; align-items: center;">'
                           f'<div style="width: 20px; height: 20px; background-color: {color_hex}; margin-right: 10px;"></div>'
                           f'<div>{class_name}</div></div>', unsafe_allow_html=True)
    
    # Fixed confidence threshold (since we removed the slider)
    conf_threshold = 0.25
    
    # Fixed debug mode setting
    debug_mode = False
    
    # Load the model with a fixed path
    model_path = "best.pt"
    try:
        with st.spinner("Loading model..."):
            model = load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure the model file exists in the same directory as this app.")
        return
    
    # File uploader for image
    uploaded_file = st.file_uploader("Upload a PV panel image", type=["jpg", "jpeg", "png"])
    
    # Camera input option (only works on devices with cameras)
    camera_input = st.camera_input("Or take a photo with your camera")
    
    # Process the uploaded image or camera image
    if uploaded_file is not None or camera_input is not None:
        input_file = uploaded_file if uploaded_file is not None else camera_input
        
        try:
            # Read the image
            file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error("Failed to decode the image. Please try another image.")
                return
                
            # Display original and processed images side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Process image and get results
            with st.spinner("Processing image..."):
                annotated_img, results, defect_counts = detect_defects(model, image, conf_threshold)
            
            with col2:
                st.subheader("Detected Defects")
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Display debug information if enabled
            if debug_mode and results is not None:
                display_debug_info(model, results)
            
            # Create and display defect summary
            create_defect_summary(defect_counts)
            
            # Add option to download the annotated image
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(annotated_img_rgb)
            
            # Create a unique filename to avoid file access issues
            unique_filename = get_unique_filename('.png')
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, unique_filename)
            
            try:
                # Save the image with the unique filename
                pil_img.save(temp_path)
                
                # Read the file for download
                with open(temp_path, 'rb') as file:
                    download_data = file.read()
                
                # Create the download button
                st.download_button(
                    label="Download Annotated Image",
                    data=download_data,
                    file_name=f"pv_defect_detection_{int(time.time())}.png",
                    mime="image/png"
                )
                
                # Try to clean up files - use a try/except to prevent errors
                try:
                    if os.path.exists(temp_path):
                        # Add a small delay to ensure file is not in use
                        time.sleep(0.1)
                        os.unlink(temp_path)
                except Exception as e:
                    # We'll continue even if we can't delete the file
                    pass
                
            except Exception as e:
                st.error(f"Error saving image: {str(e)}")
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Demo section
    st.markdown("---")
    st.subheader("About PV Panel Defects")
    
    # Explanation of the defects
    defect_explanations = {
        "ShortCircuitString": "Indicates a short circuit in a string of solar cells, causing significant power loss.",
        "ShortCircuitCell (LowPowerCell)": "Individual cells with electrical shorts, resulting in reduced power output.",
        "Crack": "Visible cracks in the solar cells, which can expand over time and reduce panel efficiency.",
        "MicroCrack": "Smaller cracks that may not be visible to the naked eye but affect performance.",
        "OtherError": "Miscellaneous defects not falling into other categories.",
        "Examined": "Areas that have been inspected and are functioning normally."
    }
    
    # Display explanations in an expander
    with st.expander("Learn about different types of PV panel defects"):
        for defect, explanation in defect_explanations.items():
            st.markdown(f"**{defect}**: {explanation}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit and YOLOv8")
    
# Run the app
if __name__ == "__main__":
    main()


# import streamlit as st
# import cv2
# import numpy as np
# import tempfile
# import os
# from PIL import Image
# import matplotlib.pyplot as plt
# from ultralytics import YOLO
# import torch

# # Set page configuration
# st.set_page_config(
#     page_title="PV Panel Defect Detection",
#     page_icon="🔍",
#     layout="wide"
# )

# # Define the defect classes
# CLASS_NAMES = [
#     "Examined", 
#     "ShortCircuitString", 
#     "ShortCircuitCell (LowPowerCell)", 
#     "Crack", 
#     "MicroCrack", 
#     "OtherError", 
#     "Contamination", 
#     "DarkSpot", 
#     "Scratch"
# ]

# # Color mapping for different defect types (BGR format for OpenCV)
# COLORS = [
#     (0, 255, 0),     # Green for Examined
#     (0, 0, 255),     # Red for ShortCircuitString
#     (255, 0, 0),     # Blue for ShortCircuitCell
#     (0, 255, 255),   # Yellow for Crack
#     (255, 0, 255),   # Magenta for MicroCrack
#     (255, 255, 0),   # Cyan for OtherError
#     (128, 0, 128),   # Purple for Contamination
#     (0, 128, 128),   # Brown for DarkSpot
#     (128, 128, 0)    # Teal for Scratch
# ]

# # Function to load the model
# @st.cache_resource
# def load_model(model_path):
#     """Load and return the YOLO model"""
#     try:
#         model = YOLO(model_path)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         st.stop()

# # Function to perform inference on an image
# def detect_defects(model, image, conf_threshold=0.25):
#     """
#     Detect defects in the given image using the model
#     Args:
#         model: Loaded YOLO model
#         image: Image to detect defects in (numpy array in BGR format)
#         conf_threshold: Confidence threshold for detections
#     Returns:
#         Annotated image, results object, and defect summary
#     """
#     # Make a copy of the image for drawing on
#     annotated_img = image.copy()
    
#     # Run inference
#     try:
#         # Convert BGR to RGB for YOLO model
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = model(rgb_image, conf=conf_threshold, verbose=False)[0]  # Get the first result
        
#         # Create a defect summary
#         defect_counts = {class_name: 0 for class_name in CLASS_NAMES}
        
#         # Check if we have any detections
#         if results is not None and results.boxes is not None and len(results.boxes) > 0:
#             boxes = results.boxes
            
#             for box in boxes:
#                 # Get box coordinates
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
#                 # Get class ID and confidence
#                 cls_id = int(box.cls.item())
#                 conf = float(box.conf.item())
                
#                 # Ensure class ID is within range
#                 if 0 <= cls_id < len(CLASS_NAMES):
#                     # Get class name and color
#                     class_name = CLASS_NAMES[cls_id]
#                     color = COLORS[cls_id]
                    
#                     # Draw rectangle
#                     cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    
#                     # Draw label
#                     label = f"{class_name}: {conf:.2f}"
#                     (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
#                     y1 = max(y1, label_height + 10)
#                     cv2.rectangle(annotated_img, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
#                     cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
#                     # Update defect count
#                     defect_counts[class_name] += 1
#                 else:
#                     st.warning(f"Detected class ID {cls_id} is out of range. Skipping this detection.")
        
#         return annotated_img, results, defect_counts
    
#     except Exception as e:
#         st.error(f"Error during detection: {str(e)}")
#         return annotated_img, None, {class_name: 0 for class_name in CLASS_NAMES}

# # Function to create a defect summary
# def create_defect_summary(defect_counts):
#     """Create a summary of detected defects"""
#     st.subheader("Defect Summary")
    
#     # Filter out classes with zero counts
#     found_defects = {cls: count for cls, count in defect_counts.items() if count > 0}
    
#     if not found_defects:
#         st.warning("No defects detected in the image. Try adjusting the confidence threshold or try a different image.")
#         return
    
#     # Display defect counts
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.write("**Defect Counts:**")
#         for cls, count in found_defects.items():
#             st.write(f"- {cls}: {count}")
    
#     with col2:
#         # Create a simple bar chart for detected defects
#         if found_defects:
#             fig, ax = plt.subplots(figsize=(10, 5))
#             classes = list(found_defects.keys())
#             counts = list(found_defects.values())
            
#             # Safely get colors for each class
#             colors = []
#             for cls in classes:
#                 try:
#                     idx = CLASS_NAMES.index(cls)
#                     colors.append(rgba_to_hex(COLORS[idx]))
#                 except ValueError:
#                     colors.append('#777777')  # Default gray if class not found
            
#             bars = ax.bar(classes, counts, color=colors)
#             ax.set_ylabel('Count')
#             ax.set_title('Defect Counts')
#             plt.xticks(rotation=45, ha='right')
#             plt.tight_layout()
#             st.pyplot(fig)

# # Helper function to convert RGB to hex color
# def rgba_to_hex(rgba):
#     """Convert RGB or RGBA to hex color string"""
#     b, g, r = rgba[:3]
#     return f'#{r:02x}{g:02x}{b:02x}'

# # Function to display model information and debug data
# def display_debug_info(model, results):
#     with st.expander("Debug Information"):
#         st.write("**Model Information:**")
#         st.write(f"Model Type: {type(model).__name__}")
#         st.write(f"Model Task: {getattr(model, 'task', 'Unknown')}")
        
#         if results is not None:
#             st.write("\n**Detection Results:**")
#             st.write(f"Result Type: {type(results).__name__}")
#             if hasattr(results, 'boxes') and results.boxes is not None:
#                 st.write(f"Number of detections: {len(results.boxes)}")
#                 if len(results.boxes) > 0:
#                     st.write("\nFirst few detections:")
#                     for i, box in enumerate(results.boxes[:3]):  # Show first 3 detections
#                         cls_id = int(box.cls.item())
#                         conf = float(box.conf.item())
#                         class_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"Unknown ({cls_id})"
#                         st.write(f"Detection {i+1}: Class={class_name}, Confidence={conf:.3f}")
#             else:
#                 st.write("No detection boxes found in results.")
#         else:
#             st.write("No results available. Model inference may have failed.")

# # Main function to create the Streamlit app
# def main():
#     # App title and description
#     st.title("PV Panel Defect Detection")
#     st.markdown("""
#     Upload an image of a photovoltaic panel to detect possible defects.
#     This application uses a YOLOv8 model trained to identify 9 different types of PV panel defects.
#     """)
    
#     # Sidebar for model settings
#     st.sidebar.header("Model Settings")
    
#     # Model path selection
#     model_path_options = {
#         "Default Model": "best.pt"
#     }
    
#     # Add custom model upload option
#     custom_model = st.sidebar.file_uploader("Upload a custom model (optional)", type=["pt"])
#     if custom_model:
#         # Save the uploaded model to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
#             tmp_file.write(custom_model.getvalue())
#             model_path = tmp_file.name
#     else:
#         selected_model = st.sidebar.selectbox("Select Model", list(model_path_options.keys()))
#         model_path = model_path_options[selected_model]
    
#     # Confidence threshold slider
#     conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.25, step=0.05)
    
#     # Toggle for debug mode
#     debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
#     # Display defect classes with their colors
#     st.sidebar.header("Defect Classes")
#     for i, class_name in enumerate(CLASS_NAMES):
#         color_hex = rgba_to_hex(COLORS[i])
#         st.sidebar.markdown(f'<div style="display: flex; align-items: center;">'
#                            f'<div style="width: 20px; height: 20px; background-color: {color_hex}; margin-right: 10px;"></div>'
#                            f'<div>{class_name}</div></div>', unsafe_allow_html=True)
    
#     # Load the model
#     try:
#         with st.spinner("Loading model..."):
#             model = load_model(model_path)
#         st.sidebar.success("Model loaded successfully!")
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         st.info("Please make sure the model file exists in the same directory as this app or upload a valid model file.")
#         return
    
#     # File uploader for image
#     uploaded_file = st.file_uploader("Upload a PV panel image", type=["jpg", "jpeg", "png"])
    
#     # Camera input option (only works on devices with cameras)
#     camera_input = st.camera_input("Or take a photo with your camera")
    
#     # Process the uploaded image or camera image
#     if uploaded_file is not None or camera_input is not None:
#         input_file = uploaded_file if uploaded_file is not None else camera_input
        
#         try:
#             # Read the image
#             file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)
#             image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
#             if image is None:
#                 st.error("Failed to decode the image. Please try another image.")
#                 return
                
#             # Display original and processed images side by side
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader("Original Image")
#                 st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
#             # Process image and get results
#             with st.spinner("Processing image..."):
#                 annotated_img, results, defect_counts = detect_defects(model, image, conf_threshold)
            
#             with col2:
#                 st.subheader("Detected Defects")
#                 st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_column_width=True)
            
#             # Display debug information if enabled
#             if debug_mode and results is not None:
#                 display_debug_info(model, results)
            
#             # Create and display defect summary
#             create_defect_summary(defect_counts)
            
#             # Add option to download the annotated image
#             annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
#             pil_img = Image.fromarray(annotated_img_rgb)
            
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
#                 pil_img.save(tmp_file.name)
#                 with open(tmp_file.name, 'rb') as file:
#                     st.download_button(
#                         label="Download Annotated Image",
#                         data=file,
#                         file_name="pv_defect_detection.png",
#                         mime="image/png"
#                     )
                
#                 # Remove the temporary file
#                 os.unlink(tmp_file.name)
            
#         except Exception as e:
#             st.error(f"Error processing image: {str(e)}")
#             if debug_mode:
#                 st.exception(e)
    
#     # Demo section
#     st.markdown("---")
#     st.subheader("About PV Panel Defects")
    
#     # Explanation of the defects
#     defect_explanations = {
#         "ShortCircuitString": "Indicates a short circuit in a string of solar cells, causing significant power loss.",
#         "ShortCircuitCell (LowPowerCell)": "Individual cells with electrical shorts, resulting in reduced power output.",
#         "Crack": "Visible cracks in the solar cells, which can expand over time and reduce panel efficiency.",
#         "MicroCrack": "Smaller cracks that may not be visible to the naked eye but affect performance.",
#         "Contamination": "Dirt, dust, bird droppings, or other materials blocking sunlight.",
#         "DarkSpot": "Areas with discoloration that indicate internal cell damage.",
#         "Scratch": "Surface damage that can affect light absorption.",
#         "OtherError": "Miscellaneous defects not falling into other categories.",
#         "Examined": "Areas that have been inspected and are functioning normally."
#     }
    
#     # Display explanations in an expander
#     with st.expander("Learn about different types of PV panel defects"):
#         for defect, explanation in defect_explanations.items():
#             st.markdown(f"**{defect}**: {explanation}")
    
#     # Footer
#     st.markdown("---")
#     st.markdown("Built with Streamlit and YOLOv8")
    
#     # Cleanup
#     if custom_model and 'model_path' in locals() and os.path.exists(model_path) and model_path.startswith(tempfile.gettempdir()):
#         try:
#             os.unlink(model_path)
#         except:
#             pass
    
# # Run the app
# if __name__ == "__main__":
#     main()
