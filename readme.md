# Byte_me: Food Segmentation and Volume Estimation with Mask R-CNN and Pinhole Camera Model

This project aims to classify food items and estimate their volume using a combination of Mask R-CNN for food segmentation and the pinhole camera model algorithm for volume estimation. The goal is to create a system that can accurately segment different types of food items from images and estimate their volumes, which can be useful for various applications such as dietary assessment, food portion estimation, and calorie counting.

## Key Features:

- **Food Segmentation:** Utilize Mask R-CNN, a state-of-the-art deep learning model, to segment food items from images. This involves identifying the pixels belonging to each food item, allowing for precise localization and segmentation.
  
- **Volume Estimation:** Implement the pinhole camera model algorithm to estimate the volume of segmented food items. By analyzing the dimensions of the food items in the image and applying geometric principles, the algorithm calculates an approximate volume.

- **Mobile App Integration:** Develop a mobile application that utilizes the phone camera to capture images of food items in real-time. Integrate the food segmentation and volume estimation algorithms into the app, allowing users to quickly analyze and estimate the volume of their meals.

- **Model Training and Evaluation:** Train the Mask R-CNN model on a custom food dataset to perform food segmentation. Evaluate the trained model's performance using metrics such as mean average precision (mAP) and intersection over union (IoU) to ensure accurate segmentation results.

- **Integration:** Integrate the segmentation and volume estimation algorithms into a unified system for practical use. This involves processing input images, performing segmentation, estimating volumes, and presenting the results in a user-friendly format.

## Project Structure:

- **`data/`:** Contains datasets for training, validation, and testing, along with annotations for food items.
  
- **`models/`:** Stores pre-trained and trained models used for food segmentation and volume estimation.

- **`notebooks/`:** Jupyter notebooks for various stages of the project, including data preprocessing, model training, evaluation, and demonstration of the segmentation and volume estimation process.

- **`src/`:** Source code directory containing Python scripts for implementing the food segmentation and volume estimation algorithms. Includes modules for data preprocessing, model architecture, and volume estimation algorithms.

- **`mobile_app/`:** Contains the source code for the mobile application, including front-end and back-end components.

- **`requirements.txt`:** Lists all Python dependencies required for running the project code. Ensure these dependencies are installed before running the code.

## Usage:

1. **Data Preparation:** Prepare your food dataset with images and corresponding annotations for training the Mask R-CNN model.

2. **Model Training:** Use the provided notebook (`model_training.ipynb`) to train the Mask R-CNN model on your custom food dataset.

3. **Volume Estimation:** Implement the pinhole camera model algorithm for estimating the volume of segmented food items. Use the provided functions in `src/volume_estimation.py` for this purpose.

4. **Mobile App Deployment:** Deploy the mobile application to your smartphone or emulator. Use the app to capture images of food items and perform real-time segmentation and volume estimation.
