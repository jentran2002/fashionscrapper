import streamlit as st
from PIL import Image
import io

def main():
    st.title("Clothing Labels Classifier")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a clothing image", type=["jpg", "jpeg", "png"])
    
    # Initialize output text area
    output_text = ""
    output_area = st.text_area("Detected Labels:", value=output_text, height=150)
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Add a button to trigger classification
        if st.button("Classify"):
            example_labels = ["dress", "shorts", "T-shirt"]
            output_text = "\n".join([f"• {label}" for label in example_labels])
            # Update text area
            st.session_state.output = output_text
            st.experimental_rerun()
    
    # Keep the output persistent
    if 'output' in st.session_state:
        output_area = st.text_area("Detected Labels:", value=st.session_state.output, height=150, key="output_area")

if __name__ == "__main__":
    main() 