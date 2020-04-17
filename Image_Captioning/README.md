# CVND---Image-Captioning-Project

## Project overview
_In this project we will try to perform image captioning on [Microsoft COCO data set](http://cocodataset.org/#home)_

## APPROACH

1. We will use pretrained CNN as an encoder which will extract feature vectors.

2. Feature vectors will then be passed on to an LSTM layer which will act as a decoder to output image captions.

## ARCHITECHTURE

<p align="center">
  <img src="images/encoder-decoder.png">
</p>

## Pretrained Weights for model
If you do not want to train the model from scratch, you can use a pretrained model. You can download the pretrained model [here](https://www.dropbox.com/sh/wu5gz3sq5nz2d6p/AAA-UWz3ed51Gv9npRGG3VWha?dl=0).The encoder-3.pkl and decoder3.pkl should be present inside `models` folder .

