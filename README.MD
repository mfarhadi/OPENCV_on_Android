# Project for EEE508 
This is the second course project for EEE508 Digital Image Processing. 

## Objectives
This  android APP is developed for the general purpose of object detection and tracking. User can select an object in the scene and track that object using the bouding box around it. A demonstration of it can be found in following section.


## Applications
This APP has great potential. It can be further improved and its possible applications include but not limited to: camera fixiation, security survilliance and even autonomous driving. 


## New Features

* Feature extraction using ORB ( Oriented FAST and Rotated BRIEF):

  * keypoint of object detected using ORB

  * keypoint of object described Using ORB

  * keypoint of scene extracted

  * keypoint of scene described

  * matches of object and scene extracted by comapring object keypoint description and scene keypoint description

Its theory can be found [here](http://www.willowgarage.com/sites/default/files/orb_final.pdf)

* Finding the best matching feature given input frame, and draw a bounding box around it:

  * select best matches

  * find matches with lest distance from each other

  * transfer matches to point in color image

  * draw rectangle that bound matches

  * create output image

The implementation of these can be found mainly in:
```
public Mat recognize(Mat aInputFrame)
```

## How to Use
Our code is tested under Android Studio 2.3.3, Ubuntu14.04, and mobile device used is under Android 7.1.1. We used OpenCV 2.3.
Steps for reproducing our result includes:

1. install all pre-requisites.
2. Open Android studio and load the project. Make sure that there is no error message for after building the project.
3. open "debug mode" on your phone and connect it to the computer
4. Run the code and select the mobile device to execute the code
5. run the the program on android device
6. find an object click on screen
7. program start following object

If anything went wrong, a detailed instructions can be found at [this tutorial](../master/assets/eee508f17_Project2_AndroidAppsWithOpenCV_Final.pdf)

## Sample Output
![acc](../master/assets/sample_output.gif)

## Team Member Contribution
- Mohammad Farhadi Bajestani: UI, bounding box (part-2), code integration
- Hope Yao: ORB (part-1)


## Acknowledgement
The team members would like to thank Seyed Alireza Golestaneh, Lina Karam and the contributors to the online resources and discussions.

