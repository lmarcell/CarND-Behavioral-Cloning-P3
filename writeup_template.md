# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_2020_05_13_10_16_24_504.jpg "Example of recorded data"
[image2]: ./images/center_2020_05_15_09_39_14_277 "Example of recorded data - recovering"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I reused the architecture designed by NVidia, since it produced better results than LeNet.

This means the following architecture:

Input image shape: 160*320*3, RGB coding
1. Preprocessing: cropping 50 pixels from the top and 20 pixels from the bottom of the picture, normalizing it and calculating 0 mean
2. Convolutional layer with 24 filters, 5*5 kernel and strides of (2,2), with relu activation function
3. Convolutional layer with 36 filters, 5*5 kernel and strides of (2,2), with relu activation function
4. Convolutional layer with 48 filters, 3*3 kernel and strides of (1,1), with relu activation function
5. Convolutional layer with 64 filters, 3*3 kernel and strides of (1,1), with relu activation function
6. Convolutional layer with 64 filters, 3*3 kernel and strides of (1,1), with relu activation function
7. Flatten layer
8. Fully connected layer with an output of 100 neurons
9. Fully connected layer with an output of 50 neurons
10. Fully connected layer with an output of 10 neurons
11. Fully connected layer with an output of 1 neurons

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I tried to introduce drop out, but it didn't bring better results, so I removed it.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 117).

#### 4. Appropriate training data

The training data was created from driving two laps in the middle of the road and about some recoveries from the side. I generated additional training data from two critical sections: entering the bridge and the big left curve after the bridge.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

First I tried to use a very simple model consisting only one convolutional layer and one fully connected layer, but I could manage to get good results with that.
After that I tried to use the LeNet architecture, it provided much better results, but it still made some mistakes in curves.
Finally I decided to use the above described architecture coming from NVidia.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 102-113) is described above.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return from difficult situations:

![alt text][image2]

Then I collected data from driving two laps reverse.

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would extend the data set in order to reach better results.

I also used the images from the side cameras by correcting their steering angles by 0.2. It helped a lot to keep the car on the road.

Later on I realized that using such a high variancy of input data doesn't enable the expected results, so finally I decided to use two data sets as inputs: driving on track 1 in the normal direction and recoveries from difficult situations, which included records from driving in critical sections as well.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by measuring the loss and the accuracy.
