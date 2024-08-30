# VGG16 Model Training with PyTorch

This repository contains a Python script for training a VGG16 model using PyTorch. The script supports custom datasets, generates confusion matrices, and plots loss and accuracy metrics.

## Table of Contents
- [Requirements](#requirements)
- [Dataset Structure](#dataset-structure)
- [Running the Script](#running-the-script)
- [Outputs](#outputs)
- [Example Dataset](#example-dataset)
- [Visualizations](#visualizations)

## Requirements
- Python 3.6+
- PyTorch
- torchvision
- numpy
- scikit-learn
- matplotlib

You can install the necessary packages using:
```bash
pip install torch torchvision numpy scikit-learn matplotlib
```

## Dataset Structure
The script expects the dataset to be organized as follows:

dataset/
  train/
     class1/
       image1.jpg
       image2.jpg
     class2/
       image3.jpg
       image4.jpg
  test/
    class1/
      image1.jpg
      image2.jpg
    class2/
      image3.jpg
      image4.jpg
Each subfolder under train/ and test/ should contain images belonging to the corresponding class.

## Running the Script
To run the script, use the following command:
```bash
python vgg16_training.py --dataroot /path/to/train --validroot /path/to/valid --datainfo /path/to/data.txt --epochs 10
```

Arguments:
--dataroot: Path to the training dataset.
--validroot: Path to the test dataset.
--datainfo: Path to the data information file (data.txt), which includes classes, number of classes, and batch size.
--epochs: Number of training epochs.


## Outputs

The script will:
-Train the VGG16 model.
-Save the best and last models based on validation accuracy.
-Generate and save confusion matrices.
-Plot and save loss and accuracy graphs for both training and validation phases.

## Example Dataset
You can use the Food-5K dataset as an example. Make sure to arrange the dataset according to the structure mentioned above.
https://archive.org/download/food-5-k/Food-5K.zip

##Visualizations
After training, the following visualizations will be generated and saved:

Confusion Matrix
The confusion matrix will be saved as confusion_matrix.png.
Example:

![confusion_matrix](https://github.com/user-attachments/assets/053ccf3a-e00a-4445-b41a-91275e81c8d2)


Loss and Accuracy Graphs
Saved as training_validation_metrics.png
Example:

![training_validation_metrics](https://github.com/user-attachments/assets/8c2e4854-419f-4558-b4e7-f10f173e1bec)


Notes
-The best model is saved based on the highest validation accuracy.
-All visualizations are saved in the directory where the script is executed.





