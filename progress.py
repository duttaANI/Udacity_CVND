"Date        concept               time   detail

8_2_20       robot localisation    2    

9_2_20       "                     2

18_2_20      cropping obj in BGR   2      opencv loads in BGR format so first convert it to RGB 

19_2_20      HSV cropping                 HSV helps in cropping independent of lighting conditions

21_2_20      day vs night                 using hsv and avg brightness to classify day vs night on a dataset

23_2_20      fourier Xform                (a)#http://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm
                                          (b)img Xform in opencv -> fourier Xform
                                          (c) opencv rgb to gray uses => Y = 0.299 R + 0.587 G + 0.114 B

24_2_20      sobel fil                    use glob library to load images
                                          #https://dsp.stackexchange.com/questions/18225/appying-a-filter-several-times-on-data?newreg=2048e1ac004642b889da96199c1cbc33


                                          (a)apply any low pass filter infinite times results in applying a gaussian filter
                                          (b)canny edge detection
                                          (c)hough transforms

25_2_20      hough Xforms                 detect circles , lines after canny edge


27_2_20      Harris Corner 
             detector
             Dialation/Erosion
             All Contours f^n

1_3_20	     ORB algo                     (a)it detects unique features like corners and creates binary feature vectors
		/-FAST algo		  (b)it is impervious to noize,rotation,illumination and scale invariant
               /-Brief algo               (c)FAST detects keypoints in an image
             DownSampling                 (d)BRIEF creates binary feature vectors and has min hardware requirement
               /-image pyramid            (e)Realtime and not for general object detection           

             HOG algo                     (a) Histogram of Oriented Gradients can do  general object detection
	     /-BlockNormalization         (b) Block=>Cell=>Pixels = Gradients
					  (c) to optimize HOG use overlapping blocks
	     
             Pytorch nn syntax            (a)#https://pytorch.org/docs/stable/nn.html

2_3_20	     Fashion MNIST                (a) Fashion-MNIST visualization using Pytorch
	                                  (b) Visualise and load ,rescale,crop using Pytorch

4_3_20       PROJECT1                     (a) creating a vgg5 model

5_3_20                                    (a) added more dropout layers , performance was a bit better
					  (b) incresed no. of channels in order 32 , 64 , 128 , 256 and perfo better than (a)
                                          (c) tried xavier initialization on (b) and perfo was better than (b) <19:37>
					  (d) TODO : change filter sizes , add more dense layers , change minibatch size
                                          (e) adding a dense layer decresed perfo than (c) <21:>
                                          (f) changing filters to 4,3,2,1 from 3,3,3,3 decresed perfo <22:22>
                                          (g) changing filters to 4,3,3,3 and batch size =16 perfo = (c) <22:50>
                                          (h) added extra conv layer but it seems to overfit perfo less than (c) <23:22> 
					  (i) added extra dense layer to (c) but perfo decresed <00:05>

6_3_20       Project1                     (j) ran (g) for 14 epochs and loss was decresing perhaps better than (c) batch size=16
 					  (k) tried to change relu to elu in (j) but perfo decreased

 					  (l) trained w/o activation f^n on last layer and perfo drastically incresed <13:30>
                                          (m) trained for 8 epochs but because of high momentum after 5 epochs perfo decresed
                   			      <17:50> also ELU was better than Relu
                 			  (n) Smooth L1 loss performed better than MSEloss in (m)

9_3_20       Project1                     (a) computerphile => voila jones face detection algo
             completed                    (b) used project1 face detection on obamas.jpg
             Voila Jones Algo                                        
             Integral Image

10_3_20      Localization                 (a) Initial_Belief -> Sense -> Move -> ( Sense -> Move )*
             (Monte Carlo)                (b) Sense increses Entropy and Move decreses it
                                          (c) Entropy=Σ(−𝑝×𝑙𝑜𝑔(𝑝))

14_3_20      2D sense->move               (a) mini project
	     (Monte Carlo)		  (b) uses of probability in AI =>Artificial Intelligence collection/AI PROGRESS/14_3_20

15_3_20      1D kalman filter             (a) initial_belief->(upadate->predict)*
					  (b) saw kalman filter dealt with only gaussian functions

16_3_20	     1D car in 2D world           (a)  
	     2D Kalman filter		  (b) 2D kalman filter =>#https://youtu.be/OelWLjfdSyw

20_3_20      omega and xi                 (a) ommega and xi are matrices for graph slam

21_3_20      Project3                     (a) implemented notebook1

22_3_20      confidence in                (a) more weight is assigned while adding values in omega and xi
	     omega and xi          

23_3_20      Project3                     (a) was able to get correct dimensions for notebook 3
                                          (b) finished Project3

26_3_20      Finished P3 		  (a) added square landmark to graph slam
					  (b) found good research paper on image captioning

27_3_20      feedforward rev^n            (a) validation set is a subset of traininng set for early stopping 
	     backprop revision		  (b) minibatch updates weights via avg(gradients) in minibatch

28_3_20      RNN revision                 (a) backpropogation in RNN is called BPTT i.e. backprop through TIME
	     BPTT

29_3_20      BPTT derivat^n
	     Gradient Clipping            (a) Gradient Clipping in RNN helps to solve Exploding gradients problem 

30_3_20      LSTM                         (a) LSTM in detail 
	     LSTM with peephole		  (b) tanh e (-1,+1) , sigmoid e (0,+1)
					  (c) peephole connections are connections of LTM to sigmoids in LSTMS
                                          (d) IMP things in rubrik
					       /-> make a validation set
					       /-> try beam search
						/-> You are not required to change anything about the encoder

31_3_20      Leraning Rate decay          (a) understood why Leraning Rate decay is imp
					  (b) small minibatch size is more likely to find global min , as large minibatch size 						      tends to get stuck at local min
             Minibatch size               (c) good start value for minibatch size - 32,64,128,256
					  (d) learning capacity of model ~= no. of hidden layers
								         ~= size of hidden layers

3_4_20	     Project2                     (a) ideas from research papers :-
						/- The most obvious way to not overfit is to ini-tialize  the  weights  of  the  CNN  component  of  our  systemto a pretrained model (e.g., on ImageNet)
						/- We  trained  all  sets  of  weights  using  stochastic  gradi-ent  descent  with  fixed  learning  rate  and  no  momentum.
						/- We used 512 dimensions for the em-beddings and the size of the LSTM memory.
						/- Descriptions were preprocessed with basic tokenization,keeping all words that appeared at least 5 times in the train-ing set
						/-  For identical decoder architectures,  us-ing more recent architectures such as GoogLeNet or Ox-ford VGG Szegedy et al. (2014), Simonyan & Zisserman(2014)  can  give  a  boost  in  performance  over  using  theAlexNet (Krizhevsky et al., 2012). 
						/- The 19-layer OxfordNet uses stacks  of 3x3 filters mean-ing the only time the feature maps decrease in size are dueto the max pooling layers.  The input image is resized sothat the shortest side is 256 dimensional with preserved as-pect ratio.   The input to the convolutional network is thecenter cropped 224x224 image. Consequently, with 4 maxpooling layers we get an output dimension of the top con-volutional layer of 14x14.  Thus in order to visualize theattention weights for the soft model, we simply upsamplethe weights by a factor of24= 16 and apply a Gaussianfilter. We note that the receptive fields of each of the 14x14units are highly overlapping
						/-  We empirically verifiedthat feeding the image at each time step as an extra inputyields inferior results, as the network can explicitly exploit noise in the image and overfits more easily





9_4_20	     Project2                     (a) MOST IMP LINK :https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8


12_4_20      Project2                     (a) I have completed training model

13_4_20      Project2                     (a) Finished all basic tasks for project

14_4_20      Project2                     (a) created json file for BLEU score






































