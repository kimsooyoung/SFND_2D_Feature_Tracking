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

### 1. Set up the loading procedure for the images
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

### 2. Implement a various bunch of detectors, which are Shi-Tomasi, HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT method.

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

### 3. Remove all keypoints outside of a bounding box around the preceding vehicle. Box parameters you should use are : cx = 535, cy = 180, w = 180, h = 150.

This job can be done by using `cv::Rect` function & `cv::Rect -> contains`
Caution!! When the element is erased, vector automatically pull the latter elements. :(
That's the reason I implement for-loop in this way. 

```c++
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle){
            // caution !! If erase element from vector, all the elements behind to erased element pulled one by one
            for (int i = keypoints.size(); i != 0; i--){
                // cout << i << endl;
                if( !vehicleRect.contains( keypoints[i].pt ) )
                    keypoints.erase(keypoints.begin() + i);
            }
            cout << "Keypoints in vehicle Rect n= " << keypoints.size() << endl;
        }
```

### 4. Implement a variety of keypoint descriptors, which are BRISK, BRIEF, ORB, FREAK, AKAZE and SIFT method.

```c++
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }else if(descriptorType.compare("BRIEF") == 0){
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }else if(descriptorType.compare("ORB") == 0){
        extractor = cv::ORB::create();
    }else if(descriptorType.compare("FREAK") == 0){
        extractor = cv::xfeatures2d::FREAK::create();
    }else if(descriptorType.compare("AKAZE") == 0){
        extractor = cv::AKAZE::create();
    }else if(descriptorType.compare("SIFT") == 0){
        extractor = cv::xfeatures2d::SIFT::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}
```

### 5. Implemente matching algorithms, Brute Force matching combined with Nearest-Neighbor selection, FLANN as an alternative to brute-force as well as the K-Nearest-Neighbor approach.

### 6. Implement the descriptor distance ratio test as a filtering method to remove bad keypoint matches.
Q: Why using Distance Ratio? 

A: ![Screenshot from 2020-02-24 15-54-11](https://user-images.githubusercontent.com/12381733/75133391-eec86500-571d-11ea-950f-016a28be57a5.png)

Imagine such case, There's few similar patterns in image, Distacne Ratio extract the distace between best 2 features in consecutive image, and we can provide the mismatch by using descriptor + distance ration combination.

```c++
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0){
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0){
        if ( (descSource.type() != CV_32F) || (descRef.type() != CV_32F) ){ // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)  
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches

        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it){
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance){
                matches.push_back((*it)[0]);
            }
        }
    }
}
```

### 7. Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size.

### 8. Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, use the BF approach with the descriptor distance ratio set to 0.8.

### 9. Log the time it takes for keypoint detection and descriptor extraction

Here's [Google Spread Sheet link](https://docs.google.com/spreadsheets/d/1H2qoxqAklakkGRWDTFXGNOmsREILkXQJLmi7ttRISDM/edit?usp=sharing) for those Tasks

![Screenshot from 2020-02-24 16-01-09](https://user-images.githubusercontent.com/12381733/75133646-e58bc800-571e-11ea-894b-bb14bd537dcd.png)

According to this report, 
1. FAST + BRIEF
2. ORB + BRIEF
3. BRIEF + BRISK
combination looks best among them.
