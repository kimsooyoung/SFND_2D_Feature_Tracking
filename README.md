# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.

## Project-overview

Final goal in this project is to find `keypoints` in every images, and also need to express them in a unique way.
It is called `descriptor`. Lastly, If there's overlap between images, we can match `descriptors` that belong to same object.

1. Set up the loading procedure for the images
In order to match `descriptors` between images, we need to hold at least 2 images in our data-structure
If `descriptor` match is done, we should get a new image and do the same process to them.
In this process, We do not have to hold the previous image on our data-structure.
So, When one process loop is done, Need to pop the previous image and get a new one
This style is reffered to "Ring Buffer"

```c++
int dataBufferSize = 2;
...
DataFrame frame;
frame.cameraImg = imgGray;
dataBuffer.push_back(frame);
if( dataBuffer.size() > dataBufferSize )
    dataBuffer.erase(dataBuffer.begin());
```

2. Implement a various bunch of detectors, which are Shi-Tomasi, HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT method.

```c++
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis){
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
	double t = (double)cv::getTickCount();
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // STUDENTS NEET TO ENTER THIS CODE (C3.2 Atom 4)

    // Look for prominent corners and instantiate keypoints
    bool nonMaxima = true;
    if (nonMaxima){
        double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
        for (size_t j = 0; j < dst_norm.rows; j++)
        {
            for (size_t i = 0; i < dst_norm.cols; i++)
            {
                int response = (int)dst_norm.at<float>(j, i);
                if (response > minResponse)
                { // only store points above a threshold

                    cv::KeyPoint newKeyPoint;
                    newKeyPoint.pt = cv::Point2f(i, j);
                    newKeyPoint.size = 2 * apertureSize;
                    newKeyPoint.response = response;

                    // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                    bool bOverlap = false;
                    for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                    {
                        double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                        if (kptOverlap > maxOverlap)
                        {
                            bOverlap = true;
                            if (newKeyPoint.response > (*it).response)
                            {                      // if overlap is >t AND response is higher for new kpt
                                *it = newKeyPoint; // replace old key point with new one
                                break;             // quit loop over keypoints
                            }
                        }
                    }
                    if (!bOverlap)
                    {                                     // only add new key point if no overlap has been found in previous NMS
                        keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                    }
                }
            } // eof loop over cols
        }     // eof loop over rows
    }else {
        for (int row = 0; row < dst_norm.rows; row++) {
            for (int col = 0; col < dst_norm.cols; col++) {
                cv::KeyPoint point(col, row, 2 * apertureSize, dst_norm.at<float>(row, col));
                keypoints.push_back(point);
            }
        }
    }

	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "Harris corner detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis){
        // visualize keypoints
        std::string windowName = "Harris Corner Detection Results";
        cv::namedWindow(windowName, 5);
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis){
    cv::Ptr<cv::FeatureDetector> detector;
    if( detectorType.compare("FAST") == 0 ){
        detector = cv::FastFeatureDetector::create();
    }else if ( detectorType.compare("BRISK") == 0 ){
        detector = cv::BRISK::create();
    }else if ( detectorType.compare("ORB") == 0 ){
        detector = cv::ORB::create();
    }else if ( detectorType.compare("AKAZE") == 0 ){
        detector = cv::AKAZE::create();
    }else if ( detectorType.compare("SIFT") == 0 ){
        detector = cv::xfeatures2d::SIFT::create();
    }

	double t = (double)cv::getTickCount();
	detector->detect(img, keypoints);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << detectorType << " detector with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

	if (bVis) {
		cv::Mat visImage = img.clone();
		cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		string windowName = detectorType + " Detector Results";
		cv::namedWindow(windowName, 6);
		imshow(windowName, visImage);
		cv::waitKey(0);
	}

}
```
