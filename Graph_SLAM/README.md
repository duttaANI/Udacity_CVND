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
  <img src="./images/omega_xi_constraints.png.png" width=75% height=75% />
</p>

1. There are two types of senor data a robot collects:
    1. motion (final robot pose - initial robot pose)
    1. measurement (landmark pose - current pose) 
    
1. This data is used as input to constraint matrices(omega and xi) and is additive in nature during input.

1. The coefficient of every component(i.e. x,y) of each pose should be positive along the diagonal of omega matrix.

<p align="center">
  <img src="./images/solution.png" width=75% height=75% />
</p>

1. Another important point to note is that input weights are directly proportional to the confidence in senor data.

1. Finally to generate map and localice robot perform the opertion:
    
<p align="center">
  <img src="./images/solution.png" width=75% height=75% />
</p>

