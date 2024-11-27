# Object Detection with Detectron2 in Aerial Imagery 

This repository contains my solution for the Sayzek Datathon competition, which focused on detecting four distinct objects in aerial imagery. The challenge presented a significantly imbalanced dataset for building the required detector. To address this challenge, I implemented a comprehensive approach combining multilabel stratified shuffle split for robust and reliable data partitioning along with repeated factor oversampling to handle class imbalance effectively. For model training, I leveraged the RetinaNet architecture with focal loss function, and enhanced the detectron2 Trainer class with an early stopping mechanism for optimal training control. While I cannot share the dataset due to competition restrictions, I have provided detailed information below regarding the required input data format to help you effectively utilize the training and prediction notebooks.

## Model Architecture

The solution utilizes **RetinaNet**, a state-of-the-art object detection architecture from Detectron2's model zoo, known for its efficiency in handling class imbalance through **Focal Loss**.

![retinanet](https://github.com/user-attachments/assets/ac04def7-d74a-46d2-b940-f7cddeeea0e0)

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


