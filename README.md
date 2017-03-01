# Vehicle Detection Project

## Overview
The project is part of the Udacity Self-driving Car Nanodegree program. The goal of the project is to use a set of methods that can detect vehicles in a video. All the code is kept in `vehicle_detection.ipynb` and without being redundant, from now on I'll just refer to the code using cell and line numbers.

## Outline
The following steps will be taken in order to achieve the goal.

1. **Preparing data**
	* Getting a good feature representation
	* Data normalization
	* Data augmentation
2. **Training a classifier**—that takes those features as input and outputs the probability of whether there is a car in the image
3. **Find cars**—define small box regions of different sizes for each frame of the video and run sub-images through the classifier to find the ones that contain cars.
4. **Video ouput**—Draw boxes that are detected to have cars on those frames and output a new video.

## Preparing data
### Getting a good feature representation
In the beginning I tried a few different combinations using the HOG features, the color histograms, and the flatten, downsized raw image data in various color spaces, to feed to the Linear SVC and the Linear Regressor (which is slightly faster and produces similar result). The result was fine and the accuracy was about 97.5 percent. However, computing these features is computationally very expensive, and there are a few situations where the classifier can't spot the white car.

Before getting more data I thought I'd give CNN a try as it's known to work best at classifying image data. I used the NVIDIA model for self driving car on the GTI and KITTI  image dataset (for [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vihecles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)) of the original RGB color space and not only the result had increased to 99.2%, but as it was no longer necessary to compute the HOG features, the predicting stage was now 50 times faster. With previous method, it would take around 50 hours on my computer to convert the video and now it took only 50 minutes!

The efficiency alone can justify a change of the method, let alone it's got even better performance. So CNN it is!

One note to this is that from my past experience, CNN seems to work better with data in HLS color space when it comes to computer vision, so I made everything HLS before feeding them in to the CNN. Also I later added some extra data from the [Udacity Label Dataset](https://github.com/udacity/self-driving-car/tree/master/annotations).

### Data normalization
I didn't know `matplotlib` library treats `PNG` images different than it treats other image types. It'd always rescale each channel to `float32` and between 0 and 1, whereas for other image types it keeps them as `uint8`. Since I've got data sources of both `PNG` and `JPG` types (and a lot more `JPG`s then `PNG`s) I decided to make everything `uint8` so that the data are consistent and then do the normalization with my `normalize_single_image` function defined in `cell [17]`.

### Data augmentation
Random brightness is done on each image in the training/cross-validation data in the keras generator so that we can (sort of) have an unlimited amount of data.

## Training the CNN
Here's the architecture of the NVIDIA model:
* Input Layer—1@64x64x3
* Convolutional Layer 1—24@30x30 (k=5, p='valid')
	* Elu—24@30x30
* Convolutional Layer 2—36@13x13 (k=5, p='valid')
	* Elu—36@13x13
* Convolutional Layer 3—48@5x5 (k=5, p='valid')
	* Elu—48@5x5
* Convolutional Layer 4—64@3x3 (k=3, p='valid')
	* Elu—64@3x3
* Convolutional Layer 5—64@1x1 (k=3, p='valid')
	* Elu—64@1x1
* Flatten Layer —64
* Fully Connected Layer 1—32
* Fully Connected Layer 2—16
* Output Layer—1

The training is rather straight forward. So I won't go into detail of that.


## Find cars
Here I'll use three examples from the `test_images/` folder to demonstrate how this works.

**test1.jpg**
![Alt text](./test1.jpg)

**test2.jpg**
![Alt text](./test2.jpg)

**test5.jpg**
![Alt text](./test5.jpg)

Here we follow the following steps:
1. **Define box regions of various sizes**
2. **Throw the parts of image contained by the boxes to the classifier and see if there are any cars**
3. **Draw the boxes with cars back to the image**
4. **Use a heat map to redraw the boxes so that cars are well defined by the rectangles**

### Define box regions of various sizes
In this project we only have to cover cars on the lanes to the right of our car. I used the `slide_window()` function in line 1 of `cell [23]` to define the box regions. We can visualize this on the test images as follows:

![Alt text](./test1.png)

Here various sizes of boxes are chosen, as cars away from our position would appear smaller on the image and larger if there are closer. In the `slide_window()` function I picked five different sizes of windows as follows: 32x32, 48x48, 64x64, 96x96, 128x128 after a few trial and error.

### Predict cars using our classifier
Using our trained CNN we can now see which boxes contain a car, and discard the ones without:

**on test1.jpg**
![Alt text](./hot_win1.png)

**on test2.jpg**
![Alt text](./hot_win2.png)

**on test3.jpg**
![Alt text](./hot_win3.png)

As we can see there are quite a few false positives, where there are boxes but no cars. We can handle this by applying a threshold to the number of overlapping boxes in the next step.

### Use heat maps to redefine the car regions
 We can use the stacked boxes to define a heat map, and the apply a threshold (here' I pick `threshold=3` to eliminate all the false positives).

We can then keep track of the maximum/minimum (x, y) coordinates of all the disconnected regions and redefine new rectangles that contain the cars. And here are the results:
![Alt text](./heat_filter.png)

We can now see  all the false positives are gone and the boxes look fine and dandy!

## Pipeline (video)
Let's see how it works with the video.

https://youtu.be/OuXRZNA6iNA

## Improvement
Although the accuracy of CNN isn't bad, the image data from our video is drawn from a different distribution where the model doesn't seem to generalize that well. One way of improving this is transfer learning. We can use, say, a trained Inception model, and use transfer learning to rebuild our classifier. Another way is to use the YOLO project which uses a completely different approach. Instead of cutting each frame into pieces and feed them into the classifier, YOLO throws the entire image into the CNN and output the object labels for the predetermined coordinates. This makes it incredibly efficient (reportedly 200 FPS).