[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"

# Facial Keypoint Detection

## Project Overview

In this project, we will try to locate 68 unique facial points on different images .

![Facial Keypoint Detection][image1]

The project consists of four Python notebooks:

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

__Notebook 4__ : Fun Filters and Keypoint Uses



### Data

1. This set of image data has been extracted from the YouTube Faces Dataset, which includes videos of people in YouTube videos. These videos have been fed through some processing steps and turned into sets of image frames containing one face and the associated keypoints.

1. This facial keypoints dataset consists of 5770 color images. All of these images are separated into either a training or a test set of data.

    1. 3462 of these images are training images, for you to use as you create a model to predict keypoints.
    1. 2308 are test images, which will be used to test the accuracy of your model.

### Dependencies
`Pytorch 0.4`

### Reference
[1] Facial Key Points Detection using DeepConvolutional Neural Network - NaimishNet (https://arxiv.org/pdf/1710.00977.pdf)
