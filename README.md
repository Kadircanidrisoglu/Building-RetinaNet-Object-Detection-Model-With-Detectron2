# Object Detection with Detectron2 in Aerial Imagery 

This repository contains my solution for the Sayzek Datathon competition, which focused on detecting four distinct objects in aerial imagery. The dataset given to us for the detector we needed to build was highly imbalanced. To solve this problem, I used repeated factor oversampling and used the focal loss function using the RetinaNet architecture for model training. I also integrated the early stopping hook into my detectron2 Trainer class. While I cannot share the dataset due to competition restrictions, I have provided detailed information below about the required input data format to help you utilize the training and prediction notebooks effectively.

## Input Data Format

### Dataset Files

- **TRAIN_DETECTRON2** / **TEST_DETECTRON2**:
  - `image data folders`

- **train_dataset_detectron2.csv** / **test_dataset_detectron2.csv**:
  - `image_id`
  - `class_name`
  - `class_id`
  - `x_min, y_min, x_max, y_max` (bounding box coordinates)

- **train_meta.csv** / **test_meta.csv**:
  - `image_id`
  - `dim0` (image height)
  - `dim1` (image width)

## Key Features

### Custom Training Pipeline

- **Data Augmentation:**  
  Implemented three distinct augmentation techniques using the Albumentations library.
  
- **Class Imbalance Handling:**
  - Repeated Factor Oversampling
  - Multilabel Stratified Shuffle Split for robust dataset partitioning

### Training Optimizations

- Lr schedular
- Early Stopping implementation  
- RetinaNet architecture from Detectron2 model zoo

## Model Architecture

The solution utilizes **RetinaNet**, a state-of-the-art object detection architecture from Detectron2's model zoo, known for its efficiency in handling class imbalance through **Focal Loss**.

