# Aerial Object Detection with Detectron2

**Implementation of a custom object detection model for the Sayzek Datathon competition using the Detectron2 framework.**  
The solution focuses on detecting 4 distinct objects in aerial imagery using advanced data augmentation and training techniques.

## Project Structure

├── TRAIN_DETECTRON2/ # Training images directory
├── TEST_DETECTRON2/ # Test images directory
├── train_dataset_detectron2.csv
├── test_dataset_detectron2.csv
├── train_meta.csv
└── test_meta.csv

## Input Data Format

### Dataset Files

- **train_dataset_detectron2.csv** / **test_dataset_detectron2.csv**:
  - `image_id`
  - `class_name`
  - `class_id`
  - `x_min, y_min, x_max, y_max` (bounding box coordinates)

### Metadata Files

- **train_meta.csv** / **test_meta.csv**:
  - `image_id`
  - `dim0` (image height)
  - `dim1` (image width)

## Key Features

### Custom Training Pipeline

- **Advanced Data Augmentation:**  
  Implemented three distinct augmentation techniques using the Albumentations library.
  
- **Class Imbalance Handling:**
  - Repeated Factor Oversampling
  - Multilabel Stratified Shuffle Split for robust dataset partitioning

### Training Optimizations

- Early Stopping implementation  
- RetinaNet architecture from Detectron2 model zoo

## Model Architecture

The solution utilizes **RetinaNet**, a state-of-the-art object detection architecture from Detectron2's model zoo, known for its efficiency in handling class imbalance through **Focal Loss**.

## Getting Started

1. **Prepare your dataset** following the specified format.
2. **Place images** in respective directories:
   - Training images in `TRAIN_DETECTRON2/`
   - Test images in `TEST_DETECTRON2/`
3. Ensure CSV files match the required schema.
