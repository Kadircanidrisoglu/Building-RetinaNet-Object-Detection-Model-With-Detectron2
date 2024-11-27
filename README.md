Aerial Object Detection with Detectron2

Implementation of a custom object detection model for the Sayzek Datathon competition using the Detectron2 framework. The solution focuses on detecting 4 distinct objects in aerial imagery using advanced data augmentation and training techniques.

Project Structure

├── TRAIN_DETECTRON2/      # Training images directory
├── TEST_DETECTRON2/       # Test images directory
├── train_dataset_detectron2.csv
├── test_dataset_detectron2.csv
├── train_meta.csv
└── test_meta.csv

Input Data Format

Dataset Files

train_dataset_detectron2.csv / test_dataset_detectron2.csv:

Column

Description

image_id

Unique identifier for each image

class_name

Object class name

class_id

Object class ID

x_min

Bounding box X-min coordinate

y_min

Bounding box Y-min coordinate

x_max

Bounding box X-max coordinate

y_max

Bounding box Y-max coordinate

Metadata Files

train_meta.csv / test_meta.csv:

Column

Description

image_id

Unique identifier for each image

dim0

Image height

dim1

Image width

Key Features

Custom Training Pipeline

Advanced Data Augmentation:

Implemented three distinct augmentation techniques using the Albumentations library.

Class Imbalance Handling:

Repeated Factor Oversampling

Multilabel Stratified Shuffle Split for robust dataset partitioning

Training Optimizations

Early Stopping implementation to avoid overfitting

RetinaNet Architecture from Detectron2 Model Zoo

Model Architecture

The solution utilizes RetinaNet, a state-of-the-art object detection architecture from the Detectron2 Model Zoo. RetinaNet is renowned for its:

Efficiency in handling class imbalance through Focal Loss

Robust performance in object detection tasks

Getting Started

Prepare Your Dataset:

Format your dataset as per the specified schema.

Place images in respective directories:

Training images: TRAIN_DETECTRON2/

Test images: TEST_DETECTRON2/

Ensure CSV Files Match the Required Schema:

train_dataset_detectron2.csv and test_dataset_detectron2.csv

train_meta.csv and test_meta.csv

Dependencies

Detectron2

Albumentations

PyTorch

pandas

numpy

Install required packages via:

pip install detectron2 albumentations torch pandas numpy

Notes

Dataset: Not included due to competition restrictions.

Focus: Custom implementations for handling class imbalance and advanced data augmentation techniques.

Use Case: Solution tailored for aerial imagery object detection tasks.

Model Training

The training pipeline includes:

Custom Data Augmentation Pipeline

Class Balancing Techniques

Early Stopping Mechanism

Stratified Data Splitting
