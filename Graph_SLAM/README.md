# Landmark Detection & Robot Tracking (GRAPH SLAM)

## Project Overview

In this project, WE'll implement GRAPH SLAM (Simultaneous Localization and Mapping) for a 2 dimensional world! 

*Below is an example of a 2D robot world with landmarks (purple x's) and the robot (a red 'o') located and found using *only* sensor and motion data collected by that robot*

<p align="center">
  <img src="./images/robot_world.png" width=50% height=50% />
</p>

The project will be broken up into three Python notebooks:

__Notebook 1__ : Robot Moving and Sensing

__Notebook 2__ : Omega and Xi, Constraints 

__Notebook 3__ : Landmark Detection and Tracking 

## APPROACH

<p align="center">
  <img src="./images/Graph_SLAM_constraints2.png" width=75% height=75% />
</p>

1. There are two types of senor data a robot collects:
    1. motion data (final robot pose - initial robot pose)
    1. measurement data(landmark pose - current pose) _The landmarks which are in range of sensor_
    
1. This data is used as input to constraint matrices(omega and xi) and is additive in nature during input.

1. The coefficient of every component(i.e. x,y) of each pose should be positive along the diagonal of omega matrix.

*THe below image is for x component and motion data*

<p align="center">
  <img src="./images/omega_xi_constraints.png" width=70% height=70% />
</p>

4. Another important point to note is that input weights are directly proportional to the confidence in senor data.

5. Finally to generate map and localice robot perform the opertion:
    
    #### mu = (omega inverse)*xi   [ matrix multiplication ]
    
<p align="center">
  <img src="./images/solution.png" width=30% height=30% />
</p>

6. Here mu is the matrix containing best possible predictions for robot poses and landmarks.

### Reference
[1] The GraphSLAMAlgorithm withApplications toLarge-Scale Mappingof Urban Structures - Sebastian Thrun (http://robots.stanford.edu/papers/thrun.graphslam.pdf)
