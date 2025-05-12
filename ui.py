import streamlit as st
from PIL import Image
import io
import zipfile
import os
import shutil
from pathlib import Path
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Define image size that matches our model
IMG_SIZE = (160, 160)

# Load the trained model at startup
@st.cache_resource
def load_classification_model():
    try:
        # Register the Cast layer as a custom object
        from tensorflow.keras.layers import Layer
        
        class CastLayer(Layer):
            def __init__(self, dtype, **kwargs):
                super(CastLayer, self).__init__(**kwargs)
                self.dtype_to_cast = dtype
                
            def call(self, inputs):
                return tf.cast(inputs, self.dtype_to_cast)
                
            def get_config(self):
                config = super(CastLayer, self).get_config()
                config.update({"dtype": self.dtype_to_cast})
                return config
        
        # Load model with custom objects
        with tf.keras.utils.custom_object_scope({'Cast': CastLayer}):
            model = load_model("clothesclassificationmodel.h5")
            print("Model loaded successfully")
            return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

CATEGORIES = ['Athletic', 'Bottoms', 'Natural', 'Tops']

CATEGORY_DESCRIPTIONS = {
    'Tops': 'T-shirts, Polo shirts, Shirts, Blazers, Hoodies, Sweaters, Outerwear',
    'Bottoms': 'Trousers, Jeans, Shorts',
    'Athletic': 'Athletic wear, Swimwear',
    'Natural': 'Linen, Natural fabrics'
}

def preprocess_image_from_bytes(image_bytes):
    """Preprocess an image from bytes for model prediction"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0 
        img_array = np.expand_dims(img_array, axis=0)  
        return img_array, img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def predict_image_from_bytes(model, image_bytes):
    """Predict the category of an image using the model"""
    img_array, img = preprocess_image_from_bytes(image_bytes)
    if img_array is not None:
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CATEGORIES[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Get confidence for all categories for visualization
        all_confidences = {CATEGORIES[i]: float(predictions[0][i]) for i in range(len(CATEGORIES))}
        
        return predicted_class, confidence, all_confidences, img
    return "Unknown", 0, {}, None

def visualize_confidences(confidences):
    """Create a horizontal bar chart of prediction confidences"""
    fig, ax = plt.subplots(figsize=(10, 5))
    categories = list(confidences.keys())
    values = list(confidences.values())
    
    # Sort by confidence value
    sorted_indices = np.argsort(values)[::-1] 
    sorted_categories = [categories[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    bars = ax.barh(sorted_categories, sorted_values, color='skyblue')
    
    # Add confidence values as text labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', va='center')
    
    ax.set_xlabel('Confidence')
    ax.set_title('Prediction Confidence by Category')
    ax.set_xlim(0, 1.1)  
    plt.tight_layout()
    
    return fig

def process_multiple_images(uploaded_images, model):
    """Process multiple uploaded images and return predictions"""
    results = []
    prediction_stats = {category: 0 for category in CATEGORIES}
    
    for img_name, img_data in uploaded_images.items():
        label, confidence, all_confidences, _ = predict_image_from_bytes(model, img_data.getvalue())
        
        prediction_stats[label] = prediction_stats.get(label, 0) + 1
        
        results.append({
            "filename": img_name,
            "category": label,
            "confidence": confidence,
            "all_confidences": all_confidences
        })
    
    return results, prediction_stats

def main():
    st.set_page_config(page_title="Clothing Classifier", page_icon="👕", layout="wide")
    
    st.title("👕 Clothing Category Classifier")
    
    # Load model
    model = load_classification_model()
    if model is None:
        st.error("Failed to load the model. Please check that the model file exists.")
        return
    
    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = []
    if "prediction_stats" not in st.session_state:
        st.session_state.prediction_stats = {}
    if "current_prediction" not in st.session_state:
        st.session_state.current_prediction = None
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app classifies clothing images into 4 categories:
        """)
        
        # Display categories with their descriptions
        for category, description in CATEGORY_DESCRIPTIONS.items():
            st.markdown(f"- **{category}**: {description}")
        
        st.markdown("""
        Upload clothing images to classify them into these categories.
        """)
        
        st.header("How to use")
        st.markdown("""
        1. Upload one or more clothing images (.jpg, .jpeg, .png)
        2. View the predicted category and confidence 
        3. Upload more images to see aggregated results
        """)
    
    # Create two tabs for single and multiple image classification
    tab1, tab2 = st.tabs(["Single Image", "Multiple Images"])
    
    with tab1:
        # Single image upload and prediction
        st.subheader("Upload a single clothing image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="single_image")
        
        if uploaded_file is not None:
            # Display the uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            with st.spinner('Classifying...'):
                label, confidence, all_confidences, _ = predict_image_from_bytes(model, uploaded_file.getvalue())
                st.session_state.current_prediction = {
                    "label": label,
                    "confidence": confidence,
                    "all_confidences": all_confidences
                }
            
            with col2:
                st.markdown(f"### Prediction: **{label}**")
                st.markdown(f"Confidence: {confidence:.2f}")
                
                # Display confidence chart
                fig = visualize_confidences(all_confidences)
                st.pyplot(fig)
    
    with tab2:
        # Multiple image upload and batch processing
        st.subheader("Upload multiple clothing images")
        uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="multiple_images")
        
        if uploaded_files:
            # Process all files when button is clicked
            if st.button("Classify All Images"):
                with st.spinner('Classifying images...'):
                    # Convert list to dictionary with filenames as keys
                    files_dict = {file.name: file for file in uploaded_files}
                    results, prediction_stats = process_multiple_images(files_dict, model)
                    
                    # Update session state
                    st.session_state.results = results
                    st.session_state.prediction_stats = prediction_stats
                    
                    st.success(f'Successfully classified {len(results)} images!')
            
            # Display results
            if st.session_state.results:
                # Create two columns for results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### Classification Results")
                    
                    # Group by category
                    results_by_category = {}
                    for result in st.session_state.results:
                        category = result["category"]
                        if category not in results_by_category:
                            results_by_category[category] = []
                        results_by_category[category].append(result)
                    
                    # Display in expandable sections
                    for category, items in results_by_category.items():
                        with st.expander(f"📁 {category} ({len(items)} images)"):
                            for item in items:
                                st.text(f"📄 {item['filename']} (Confidence: {item['confidence']:.2f})")
                
                with col2:
                    # Display prediction distribution
                    st.markdown("### Category Distribution")
                    if st.session_state.prediction_stats:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        categories = list(st.session_state.prediction_stats.keys())
                        counts = list(st.session_state.prediction_stats.values())
                        
                        # Create horizontal bar chart
                        bars = ax.barh(categories, counts, color='skyblue')
                        
                        # Add count labels
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                                    f'{width:.0f}', va='center')
                        
                        ax.set_xlabel('Number of Images')
                        ax.set_title('Classification Distribution')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display category stats as a table
                        st.markdown("### Category Counts")
                        data = {"Category": [], "Count": [], "Percentage": []}
                        total = sum(st.session_state.prediction_stats.values())
                        
                        for cat, count in st.session_state.prediction_stats.items():
                            data["Category"].append(cat)
                            data["Count"].append(count)
                            data["Percentage"].append(f"{(count/total)*100:.1f}%")
                        
                        st.table(data)

if __name__ == "__main__":
    main()
