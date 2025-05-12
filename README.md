Clothing Classification Model
Project Overview
This project implements a deep learning model to automatically classify clothing items into four categories: Tops, Bottoms, Athletic, and Natural. Using transfer learning with MobileNetV2 as the base architecture, the model achieves approximately 79% validation accuracy despite significant class imbalance and limited training data.
Dataset
Source: Collection of clothing images with 1,664 total images after processing
Distribution:
Tops: 1,052 images (63.2%)
Bottoms: 406 images (24.4%)
Athletic: 182 images (10.9%)
Natural: 24 images (1.4%)
Image size: 160x160 pixels (RGB)
Preprocessing: Normalized pixel values to [0,1]
Model Architecture
Base model: MobileNetV2 pre-trained on ImageNet (weights frozen)
Additional layers:
Global Average Pooling
Dense layer (128 units, ReLU activation)
Dropout (0.5)
Output layer (softmax activation)
Optimization: Adam optimizer with learning rate 0.001
Loss function: Sparse categorical cross-entropy
Technical Features
Mixed precision training using float16 policy for faster computation
Data augmentation with rotation, shifting, flipping, and zoom
Early stopping with patience=3 to prevent overfitting
Transfer learning approach to leverage pre-trained weights
