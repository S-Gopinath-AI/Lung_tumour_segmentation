Lung Tumour Classification and Explainability using DenseNet121

Overview:
This project presents a deep learning–based system for automated lung tumour classification from CT scan images. A transfer learning approach using DenseNet121 is applied to classify lung images into three categories:

Benign
Malignant
Normal

To enhance model transparency, Grad-CAM (Gradient-weighted Class Activation Mapping) is implemented to visualize the regions of the image that influence the model’s predictions.

Key Features:
Transfer Learning using DenseNet121 (ImageNet pretrained)
Class imbalance handling using class weights
Data augmentation for minority class improvement
Early stopping to prevent overfitting

Performance evaluation using confusion matrix and classification report

Grad-CAM visualization for explainable AI

Model Architecture:
Base Model: DenseNet121 (pretrained on ImageNet)
Last 50 layers unfrozen for fine-tuning
Global Average Pooling
Dense layer (256 units, ReLU activation)
Dropout (0.3)
Output layer (Softmax, 3 classes)
Loss Function: Categorical Crossentropy
Optimizer: Adam
Evaluation Metric: Accuracy

Project Structure
Lung_tumour_segmentation/
│
├── DenseNet.py              # Model training and evaluation
├── GradCam.py               # Grad-CAM visualization
├── lung_densenet_model.keras
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
└── README.md

Dataset Structure:
The dataset is organized into three classes:
train/
    ├── Bengin cases
    ├── Malignant cases
    └── Normal cases

Images are resized to 224x224 pixels and normalized using rescaling (1./255).

Installation

Clone the repository:
git clone https://github.com/S-Gopinath-AI/Lung_tumour_segmentation.git
cd Lung_tumour_segmentation

Install dependencies:

pip install -r requirements.txt

If requirements.txt is not available:

pip install tensorflow numpy opencv-python matplotlib scikit-learn
Training the Model

Run:
python DenseNet.py

This will:
Perform data augmentation (if needed)
Train the DenseNet121 model
Evaluate performance on the test dataset
Generate classification report and confusion matrix
Save the trained model as lung_densenet_model.keras
Running Grad-CAM Visualization

After training, run:
python GradCam.py

This will:
Load the trained model
Predict the class of a test image
Generate a Grad-CAM heatmap
Overlay heatmap on the original CT scan

Evaluation Metrics
The model evaluation includes:

Test Accuracy
Test Loss
Confusion Matrix
Precision, Recall, and F1-score (via classification report)
Explainable AI (Grad-CAM)

Grad-CAM highlights the important regions in the CT scan that contribute to the classification decision. This improves model interpretability and supports clinical reliability.

Future Improvements:

Implement full segmentation instead of classification
Deploy as a web application using Streamlit
Add Dice Score and IoU metrics
Use U-Net for pixel-level tumour segmentation
Add cross-validation for stronger generalization

Author
Gopinath S

Deep Learning and AI Enthusiast
GitHub: https://github.com/S-Gopinath-AI

Deep Learning and AI Enthusiast

GitHub: https://github.com/S-Gopinath-AI
