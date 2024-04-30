Prototyping
│
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── annotations/
│   ├── val/
│   │   ├── images/
│   │   └── annotations/
│   └── test/
│       └── images/
│
├── models/
│   ├── pretrained_mask_rcnn.h5
│   └── trained_food_segmentation_classification_model.h5
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── model_evaluation.ipynb
│   └── inference_demo.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   └── inference.py
│
└── requirements.txt

| Directory       | Description                                                                                                   |
|-----------------|---------------------------------------------------------------------------------------------------------------|
| `data/`         | Contains subdirectories for organizing your data into different sets.                                         |
|                 | - `train/`: Contains training data, including images and their corresponding annotations.                      |
|                 | - `val/`: Holds validation data used for evaluating the model during training. Similar to the training set.  |
|                 | - `test/`: Contains images for testing the trained model's performance. Typically doesn't include annotations. |
| `models/`       | Used for storing model weights.                                                                              |
|                 | - `pretrained_mask_rcnn.h5`: Pre-trained weights for the Mask R-CNN model.                                    |
|                 | - `trained_food_segmentation_classification_model.h5`: Weights of your trained model after training.          |
| `notebooks/`    | Contains Jupyter notebooks for various project stages.                                                        |
|                 | - `data_preprocessing.ipynb`: Details preprocessing steps for data.                                           |
|                 | - `model_training.ipynb`: Code for training the Mask R-CNN model.                                             |
|                 | - `model_evaluation.ipynb`: Evaluates the model's performance on validation data.                              |
|                 | - `inference_demo.ipynb`: Demonstrates model inference on new images.                                          |
| `src/`          | Source code directory for project tasks.                                                                      |
|                 | - `data_preprocessing.py`: Contains functions and classes for preprocessing data.                             |
|                 | - `model.py`: Defines the architecture of the Mask R-CNN model for food segmentation and classification.     |
|                 | - `inference.py`: Contains functions for running inference with the trained model on new images.             |
| `requirements.txt` | Lists all Python dependencies required for the project.                                                       |

