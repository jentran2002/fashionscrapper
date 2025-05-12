# Clothing Classification Model

## Project Overview
This project implements a deep learning model to automatically classify clothing items into four categories: Tops, Bottoms, Athletic, and Natural. Using transfer learning with MobileNetV2 as the base architecture, the model achieves approximately 79% validation accuracy despite significant class imbalance and limited training data.

## Dataset
- **Source**: Collection of clothing images with 1,664 total images after processing
- **Distribution**:
  - Tops: 1,052 images (63.2%)
  - Bottoms: 406 images (24.4%)
  - Athletic: 182 images (10.9%)
  - Natural: 24 images (1.4%)
- **Image size**: 160x160 pixels (RGB)
- **Preprocessing**: Normalized pixel values to [0,1]

## Model Architecture
- **Base model**: MobileNetV2 pre-trained on ImageNet (weights frozen)
- **Additional layers**:
  - Global Average Pooling
  - Dense layer (128 units, ReLU activation)
  - Dropout (0.5)
  - Output layer (softmax activation)
- **Optimization**: Adam optimizer with learning rate 0.001
- **Loss function**: Sparse categorical cross-entropy

## Technical Features
- **Mixed precision training** using float16 policy for faster computation
- **Data augmentation** with rotation, shifting, flipping, and zoom
- **Early stopping** with patience=3 to prevent overfitting
- **Transfer learning** approach to leverage pre-trained weights

## Requirements
```
tensorflow>=2.9.0
numpy
pandas
matplotlib
scikit-learn
pillow
tqdm
```

## Usage

### Installation
```bash
# Clone this repository
git clone https://github.com/username/clothing-classification.git
cd clothing-classification

# Install dependencies
pip install -r requirements.txt
```

### Training the Model
The model training process is contained in `clothesclassificationmodel.ipynb`. To retrain:
1. Open the notebook in Jupyter or Google Colab
2. Adjust the `IMAGES_DIR` path to point to your dataset
3. Run all cells to train and evaluate the model

### Using the Trained Model
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the model
model = tf.keras.models.load_model("clothesclassificationmodel.h5")

# Categories
categories = ['Athletic', 'Bottoms', 'Natural', 'Tops']

# Function to predict an image
def predict_image(image_path):
    img = load_img(image_path, target_size=(160, 160))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return {
        'category': categories[predicted_class],
        'confidence': float(confidence),
        'all_probabilities': {categories[i]: float(predictions[0][i]) for i in range(len(categories))}
    }

# Example usage
result = predict_image("path/to/your/image.jpg")
print(f"Predicted category: {result['category']} with {result['confidence']:.2f} confidence")
```

## Performance
- **Overall accuracy**: 78.9% on validation set
- **Per-class metrics**:
  ```
  Classification Report:
              precision    recall  f1-score   support

    Athletic       0.91      0.58      0.71        36
     Bottoms       0.78      0.49      0.61        81
     Natural       0.00      0.00      0.00         5
        Tops       0.78      0.96      0.86       211
  ```
- The model performs well on the Tops category (96% recall) but struggles with the Natural category due to extreme class imbalance

## Future Improvements
- Collect more training data, especially for underrepresented categories
- Implement class-balanced focal loss to address class imbalance
- Explore progressive layer unfreezing techniques
- Test alternative architectures such as EfficientNet-B0
- Implement test-time augmentation for improved inference

## Web Application
This model has been integrated into a Streamlit web application that allows users to upload images and receive classifications. The application code is available in the `ui.py` file.

## License
[MIT License](LICENSE)

## Acknowledgments
- MobileNetV2 architecture by Google Research
- TensorFlow and Keras for the deep learning framework
