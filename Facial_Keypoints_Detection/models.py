## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1 = nn.Conv2d(1, 32, (5,5)) #X = nn.Conv2D( 1, 1, 3, 1, 1) #  ( input_c, output_c, k_size, stride, padding ), k_size can be (3,3) or 3 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128,256,3 ) 
        #self.conv5 = nn.Conv2d(256,256,1 ) 
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.3)
        self.drop4 = nn.Dropout(0.4)
        self.drop5 = nn.Dropout(0.5)
        #self.drop6 = nn.Dropout(0.6)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.fc1 = nn.Linear(in_features =12*12*256, out_features =1024)  # !!!!!! CHANGE 12*12*256 = 36864 acc to size after flattening
        #self.act = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features =1024,out_features =512)
        
        self.fc3 = nn.Linear(512,136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.drop1(self.pool(F.elu(self.conv1(x))))# (None, 224, 224,  1) => (None, 220, 220, 32) => (None, 110, 110, 32)
        
        x = self.drop2(self.pool(F.elu(self.conv2(x))))# (None, 110, 110, 32) => (None, 108, 108, 64) => (None, 54 , 54, 64)
        
        x = self.drop3(self.pool(F.elu(self.conv3(x))))# (None, 54, 54, 64)   => (None, 52, 52, 128)  => (None, 26, 26, 128)
        
        x = self.drop4(self.pool(F.elu(self.conv4(x))))# (None, 26, 26, 128)  => (None, 24, 24, 256)  => (None, 12, 12, 256)
        
        #x = self.drop4(self.pool(F.relu(self.conv5(x))))# (None, 12, 12, 256) => (None, 12, 12, 256) => (None, 6, 6, 256)
        
        x = x.view( -1, 12*12*256)    # !!!!!! CHANGE 12*12*256 acc to size after flattening
        """
        OR x = x.view( x.size(0),-1)
        """
        #print(" x.size(0) : ", x.size(0))
        #print("x.shape",x.shape)
        x = self.drop5(F.relu(self.fc1(x)))               # (None,12*12*256) => (None,1000)
        
        x = self.drop5(F.relu(self.fc2(x)))              # (None,1000) => (None,136)
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
