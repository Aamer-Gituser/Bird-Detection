
Bird-Detection-2 - v1 2-Class_dataset
==============================

This dataset was exported via roboflow.com on December 8, 2025 at 12:12 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 18667 images.
Objects-71ML are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Fit within)

The following augmentation was applied to create 2 versions of each source image:
* 50% probability of horizontal flip
* Random rotation of between -9 and +9 degrees
* Random brigthness adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 1.3 pixels
* Salt and pepper noise was applied to 0.65 percent of pixels


