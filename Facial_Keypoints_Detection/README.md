[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection" 

# Facial Keypoint Detection

[![Udacity Computer Vision Nanodegree](http://tugan0329.bitbucket.io/imgs/github/cvnd.svg)](https://www.udacity.com/course/computer-vision-nanodegree--nd891)

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
    
### Architechture

Net(
  (conv1): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
  (conv4): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
  (drop1): Dropout(p=0.1)
  (drop2): Dropout(p=0.2)
  (drop3): Dropout(p=0.3)
  (drop4): Dropout(p=0.4)
  (drop5): Dropout(p=0.5)
  (fc1): Linear(in_features=36864, out_features=1024, bias=True)
  (fc2): Linear(in_features=1024, out_features=512, bias=True)
  (fc3): Linear(in_features=512, out_features=136, bias=True)
)

| Layer               	| Details                                                                                          	|
|---------------------	|--------------------------------------------------------------------------------------------------	|
| Input               	| size : (224, 224, 1)                                                                             	|
| Conv 1              	| # filters : 32;  kernel size : (5 x 5);  stride : (1 x 1);  <br>padding : 0;   activation : ELU          	|
| Max Pooling         	| kernel size : (2 x 2);  stride : (2 x 2);  padding : 0 (VALID)                                     	|
| Dropout             	| probability : 0.1                                                                               	|
| Conv 2              	| # filters : 64;  kernel size : (3 x 3);  stride : (1 x 1);  <br>padding : 2 (SAME);   activation : ELU 	|                                                                              	|
| Max Pooling         	| kernel size : (3 x 3);  stride : (2 x 2);  padding : 0 (VALID)                                     	|
| Dropout             	| probability : 0.2                                                                               	|
| Conv 3              	| # filters : 128;  kernel size : (3 x 3); stride : (1 x 1); <br>padding : 1 (SAME); activation : ELU   	|                                                                                	|
| Dropout             	| probability : 0.3                                                                               	|
| Conv 4              	| # filters : 256;  kernel size : (3 x 3);  stride : (1 x 1);  <br>padding : 1 (SAME);   activation : ELU  	|                                                                                	|
| Dropout             	| probability : 0.4                |                                                                                                                                  
| Flatten             | (12 x 12 x 256) => 36864               |
| Fully Connected 1   | # neurons : 1024; activation : RELU   |
| Dropout             | probability : 0.5                   |
| Fully Connected 2   | # neurons : 512; activation : RELU   |
| Dropout             | probability : 0.5                   |
| probability : 0.6   | # neurons : 136; activation : None |
| Output              | size : (136 x 1)                    |

### Dependencies
`Pytorch 0.4`

### Reference
[1] Facial Key Points Detection using DeepConvolutional Neural Network - NaimishNet (https://arxiv.org/pdf/1710.00977.pdf)
