import streamlit as st
from PIL import Image
import io
import zipfile
import os
import shutil
from pathlib import Path
import tempfile

def process_images_from_zip(zip_file):
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    
    file_structure = {}  # Store directory structure as dict
    
    try:
        # Extract uploaded zip file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Process all images
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    
                    # TODO: Replace with actual model prediction
                    # Using placeholder label for now
                    label = "dress"  # Will be replaced with model prediction
                    
                    # Create directory for this label
                    label_dir = os.path.join(output_dir, label)
                    os.makedirs(label_dir, exist_ok=True)
                    
                    # Copy image to corresponding directory
                    shutil.copy2(img_path, os.path.join(label_dir, file))
                    
                    # Add to file structure
                    if label not in file_structure:
                        file_structure[label] = []
                    file_structure[label].append(file)
        
        # Create result zip file
        result_zip = os.path.join(temp_dir, 'classified_images.zip')
        with zipfile.ZipFile(result_zip, 'w') as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
        
        with open(result_zip, 'rb') as f:
            return f.read(), file_structure
    
    finally:
        # Cleanup temporary directories
        shutil.rmtree(temp_dir)
        shutil.rmtree(output_dir)

def main():
    # Title with icon
    st.title("👕 Clothing Labels Classifier")
    
    # Initialize session state
    if "file_structure" not in st.session_state:
        st.session_state.file_structure = {}
    if "processed_zip" not in st.session_state:
        st.session_state.processed_zip = None
    
    # File uploader for zip file
    uploaded_file = st.file_uploader("Upload a ZIP file containing clothing images", type=["zip"])
    
    # Create fixed containers for results
    results_container = st.container()
    button_container = st.container()
    
    # Process uploaded file
    if uploaded_file is not None:
        with st.spinner('Processing images...'):
            # Process the zip file
            zip_data, file_structure = process_images_from_zip(uploaded_file)
            
            # Update session state
            st.session_state.file_structure = file_structure
            st.session_state.processed_zip = zip_data
            
            st.success('Classification complete! Click the download button to get your classified images.')
    
    # Display results in fixed containers
    with results_container:
        st.markdown("### Classification Results")
        if st.session_state.file_structure:
            for label, files in st.session_state.file_structure.items():
                with st.expander(f"📁 {label} ({len(files)} images)"):
                    for file in files:
                        st.text(f"📄 {file}")
    
    with button_container:
        if st.session_state.processed_zip is not None:
            st.download_button(
                label="Download Classified Images",
                data=st.session_state.processed_zip,
                file_name="classified_images.zip",
                mime="application/zip"
            )

if __name__ == "__main__":
    main() 