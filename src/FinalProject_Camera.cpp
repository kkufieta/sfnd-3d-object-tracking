/* INCLUDES FOR THIS PROJECT */
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <sstream>
#include <vector>

#include "camFusion.hpp"
#include "dataStructures.h"
#include "lidarData.hpp"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"

using namespace std;

void get_calibration_data(cv::Mat &RT, cv::Mat &R_rect_00, cv::Mat &P_rect_00) {
  // calibration data for camera and lidar

  // rotation matrix and translation vector
  RT.at<double>(0, 0) = 7.533745e-03;
  RT.at<double>(0, 1) = -9.999714e-01;
  RT.at<double>(0, 2) = -6.166020e-04;
  RT.at<double>(0, 3) = -4.069766e-03;
  RT.at<double>(1, 0) = 1.480249e-02;
  RT.at<double>(1, 1) = 7.280733e-04;
  RT.at<double>(1, 2) = -9.998902e-01;
  RT.at<double>(1, 3) = -7.631618e-02;
  RT.at<double>(2, 0) = 9.998621e-01;
  RT.at<double>(2, 1) = 7.523790e-03;
  RT.at<double>(2, 2) = 1.480755e-02;
  RT.at<double>(2, 3) = -2.717806e-01;
  RT.at<double>(3, 0) = 0.0;
  RT.at<double>(3, 1) = 0.0;
  RT.at<double>(3, 2) = 0.0;
  RT.at<double>(3, 3) = 1.0;

  // 3x3 rectifying rotation to make image planes co-planar
  R_rect_00.at<double>(0, 0) = 9.999239e-01;
  R_rect_00.at<double>(0, 1) = 9.837760e-03;
  R_rect_00.at<double>(0, 2) = -7.445048e-03;
  R_rect_00.at<double>(0, 3) = 0.0;
  R_rect_00.at<double>(1, 0) = -9.869795e-03;
  R_rect_00.at<double>(1, 1) = 9.999421e-01;
  R_rect_00.at<double>(1, 2) = -4.278459e-03;
  R_rect_00.at<double>(1, 3) = 0.0;
  R_rect_00.at<double>(2, 0) = 7.402527e-03;
  R_rect_00.at<double>(2, 1) = 4.351614e-03;
  R_rect_00.at<double>(2, 2) = 9.999631e-01;
  R_rect_00.at<double>(2, 3) = 0.0;
  R_rect_00.at<double>(3, 0) = 0;
  R_rect_00.at<double>(3, 1) = 0;
  R_rect_00.at<double>(3, 2) = 0;
  R_rect_00.at<double>(3, 3) = 1;

  // 3x4 projection matrix after rectification
  P_rect_00.at<double>(0, 0) = 7.215377e+02;
  P_rect_00.at<double>(0, 1) = 0.000000e+00;
  P_rect_00.at<double>(0, 2) = 6.095593e+02;
  P_rect_00.at<double>(0, 3) = 0.000000e+00;
  P_rect_00.at<double>(1, 0) = 0.000000e+00;
  P_rect_00.at<double>(1, 1) = 7.215377e+02;
  P_rect_00.at<double>(1, 2) = 1.728540e+02;
  P_rect_00.at<double>(1, 3) = 0.000000e+00;
  P_rect_00.at<double>(2, 0) = 0.000000e+00;
  P_rect_00.at<double>(2, 1) = 0.000000e+00;
  P_rect_00.at<double>(2, 2) = 1.000000e+00;
  P_rect_00.at<double>(2, 3) = 0.000000e+00;
}

void object_tracking(string detectorType, string descriptorType,
                     vector<double> &detectTimes, vector<double> &describeTimes,
                     vector<double> &totalTimes, vector<double> &TTCsCamera,
                     vector<double> &TTCsLidar, bool bVis = false,
                     bool bDebug = false, bool bSafe = false) {
  // data location
  string dataPath = "../";

  // camera
  string imgBasePath = dataPath + "images/";
  string imgPrefix =
      "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
  string imgFileType = ".png";
  int imgStartIndex = 0; // first file index to load (assumes Lidar and camera
                         // names have identical naming convention)
  int imgEndIndex = 30;  // last file index to load
  int imgStepWidth = 1;
  // no. of digits which make up the file index (e.g. img-0001.png)
  int imgFillWidth = 4;

  // object detection
  string yoloBasePath = dataPath + "dat/yolo/";
  string yoloClassesFile = yoloBasePath + "coco.names";
  string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
  string yoloModelWeights = yoloBasePath + "yolov3.weights";

  // Lidar
  string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
  string lidarFileType = ".bin";

  // calibration data for camera and lidar
  // 3x4 projection matrix after rectification
  cv::Mat P_rect_00(3, 4, cv::DataType<double>::type);
  // 3x3 rectifying rotation to make image planes co-planar
  cv::Mat R_rect_00(4, 4, cv::DataType<double>::type);
  // rotation matrix and translation vector
  cv::Mat RT(4, 4, cv::DataType<double>::type);
  get_calibration_data(RT, R_rect_00, P_rect_00);

  // misc
  // frames per second for Lidar and camera
  double sensorFrameRate = 10.0 / imgStepWidth;
  int dataBufferSize = 2;       // no. of images which are held in memory (ring
                                // buffer) at the same time
  vector<DataFrame> dataBuffer; // list of data frames which are held in memory
                                // at the same time

  cv::Size topviewImageSize = cv::Size(2000, 2000);
  string imgFullFilename;
  // Create and initialize the VideoWriter object
  // Get the image size for the videowriter object
  // assemble filenames for current index
  ostringstream imgNumber;
  imgNumber << setfill('0') << setw(imgFillWidth) << 0;
  imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

  // load image from file
  cv::Mat img = cv::imread(imgFullFilename);
  cv::VideoWriter detObjVideoWriter("../dat/img/detectedObjectsYolo.avi",
                                    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                    sensorFrameRate, img.size(), true);
  cv::VideoWriter kptsVideoWriter("../dat/img/keypoints-" + detectorType + "-" +
                                      descriptorType + ".avi",
                                  cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                  sensorFrameRate, img.size(), true);
  cv::VideoWriter ttcVideoWriter("../dat/img/ttc-" + detectorType + "-" +
                                     descriptorType + ".avi",
                                 cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                 sensorFrameRate, img.size(), true);
  cv::VideoWriter topviewVideoWriter(
      "../dat/img/topviewlidar.avi",
      cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), sensorFrameRate,
      topviewImageSize, true);
  if (detObjVideoWriter.isOpened() == false ||
      kptsVideoWriter.isOpened() == false ||
      ttcVideoWriter.isOpened() == false ||
      topviewVideoWriter.isOpened() == false) {
    cout << "Cannot save the videos to a file" << endl;
    return;
  }
  /* MAIN LOOP OVER ALL IMAGES */
  double t_total;
  for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex;
       imgIndex += imgStepWidth) {
    t_total = (double)cv::getTickCount();
    cout << "Processing Frame " << imgIndex << endl;
    /* LOAD IMAGE INTO BUFFER */

    // assemble filenames for current index
    ostringstream imgNumber;
    imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
    imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

    // load image from file
    cv::Mat img = cv::imread(imgFullFilename);

    // ring buffer: Keep only the last two frames
    if (dataBuffer.size() >= dataBufferSize) {
      dataBuffer.erase(dataBuffer.begin());
    }

    // push image into data frame buffer
    DataFrame frame;
    frame.cameraImg = img;
    dataBuffer.push_back(frame);

    if (bDebug)
      cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

    /* DETECT & CLASSIFY OBJECTS using YOLO */

    float confThreshold = 0.2;
    float nmsThreshold = 0.4;
    cv::Mat visObjectsImg;
    detectObjects((dataBuffer.end() - 1)->cameraImg,
                  (dataBuffer.end() - 1)->boundingBoxes, confThreshold,
                  nmsThreshold, yoloBasePath, yoloClassesFile,
                  yoloModelConfiguration, yoloModelWeights, visObjectsImg, bVis,
                  bSafe);
    if (bSafe) {
      detObjVideoWriter.write(visObjectsImg);
    }

    if (bDebug)
      cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;

    /* CROP LIDAR POINTS */

    // load 3D Lidar points from file
    string lidarFullFilename =
        imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
    std::vector<LidarPoint> lidarPoints;
    loadLidarFromFile(lidarPoints, lidarFullFilename);

    // remove Lidar points based on distance properties
    float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0,
          minR = 0.1; // focus on ego lane
    cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

    (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

    if (bDebug)
      cout << "#3 : CROP LIDAR POINTS done" << endl;

    /* CLUSTER LIDAR POINT CLOUD */

    // associate Lidar points with camera-based ROI
    float shrinkFactor =
        0.15; // shrinks each bounding box by the given percentage to avoid 3D
              // object merging at the edges of an ROI
    clusterLidarWithROI((dataBuffer.end() - 1)->boundingBoxes,
                        (dataBuffer.end() - 1)->lidarPoints, shrinkFactor,
                        P_rect_00, R_rect_00, RT);

    // Visualize 3D objects
    if (bVis || bSafe) {
      // create topview image

      cv::Mat topviewImg(topviewImageSize, CV_8UC3, cv::Scalar(255, 255, 255));
      show3DObjects((dataBuffer.end() - 1)->boundingBoxes, cv::Size(4.0, 20.0),
                    topviewImageSize, topviewImg, bVis);
      topviewVideoWriter.write(topviewImg);
    }

    if (bDebug)
      cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;

    /* DETECT IMAGE KEYPOINTS */

    // convert current image to grayscale
    cv::Mat imgGray;
    cv::cvtColor((dataBuffer.end() - 1)->cameraImg, imgGray,
                 cv::COLOR_BGR2GRAY);

    // extract 2D keypoints from current image
    // create empty feature list for current image
    vector<cv::KeyPoint> keypoints;

    // Detector types:
    // -> Gradient Based: HARRIS, SHITOMASI, SIFT
    // -> Binary: BRISK, ORB, AKAZE, FAST
    double detectTime;
    if (detectorType.compare("SHITOMASI") == 0) {
      detectTime = detKeypointsShiTomasi(keypoints, imgGray, bVis);
    } else if (detectorType.compare("HARRIS") == 0) {
      detectTime = detKeypointsHarris(keypoints, imgGray, bVis);
    } else { // SIFT, BRISK, ORB, AKAZE, FAST
      detectTime = detKeypointsModern(keypoints, imgGray, detectorType, bVis);
    }
    detectTimes.push_back(detectTime);

    // cout << "Number total keypoints: " << keypoints.size() << endl;
    vector<cv::KeyPoint> reducedKeypoints;
    for (cv::KeyPoint kp : keypoints) {
      for (BoundingBox bb : (dataBuffer.end() - 1)->boundingBoxes) {
        if (bb.roi.contains(kp.pt)) {
          // bb.keypoints.push_back(kp); Done in clusterKptMatchesWithROI in
          // camFusion.cpp
          reducedKeypoints.push_back(kp);
          break;
        }
      }
    }
    keypoints = reducedKeypoints;
    // cout << "Number reduced keypoints: " << keypoints.size() << endl;

    if (bVis || bSafe) {
      // visualize results
      cv::Mat vis_image = img.clone();
      cv::drawKeypoints(img, keypoints, vis_image);
      kptsVideoWriter.write(vis_image);
      if (bVis) {
        string windowName = "Keypoints cropped to bounding boxes";
        cv::namedWindow(windowName, 6);
        cv::imshow(windowName, vis_image);
        cv::waitKey(0);
      }
    }

    // optional : limit number of keypoints (helpful for debugging and learning)
    bool bLimitKpts = false;
    if (bLimitKpts) {
      int maxKeypoints = 50;

      if (detectorType.compare("ORB") ==
          0) { // there is no response info, so keep the first 50 as they are
               // sorted in descending quality order
        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
      }
      cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
      if (bDebug)
        cout << " NOTE: Keypoints have been limited!" << endl;
    }

    // push keypoints and descriptor for current frame to end of data buffer
    (dataBuffer.end() - 1)->keypoints = keypoints;

    if (bDebug)
      cout << "#5 : DETECT KEYPOINTS done" << endl;

    /* EXTRACT KEYPOINT DESCRIPTORS */

    cv::Mat descriptors;
    double describeTime = descKeypoints((dataBuffer.end() - 1)->keypoints,
                                        (dataBuffer.end() - 1)->cameraImg,
                                        descriptors, descriptorType);
    describeTimes.push_back(describeTime);

    // push descriptors for current frame to end of data buffer
    (dataBuffer.end() - 1)->descriptors = descriptors;

    if (bDebug)
      cout << "#6 : EXTRACT DESCRIPTORS done" << endl;

    if (dataBuffer.size() >
        1) // wait until at least two images have been processed
    {

      /* MATCH KEYPOINT DESCRIPTORS */

      vector<cv::DMatch> matches;
      string matcherType = "MAT_BF"; // MAT_BF, MAT_FLANN
      string descriptorType_ = descriptorType.compare("SIFT") == 0
                                   ? "DES_HOG"
                                   : "DES_BINARY"; // DES_BINARY, DES_HOG
      string selectorType = "SEL_NN";              // SEL_NN, SEL_KNN

      matchDescriptors((dataBuffer.end() - 2)->keypoints,
                       (dataBuffer.end() - 1)->keypoints,
                       (dataBuffer.end() - 2)->descriptors,
                       (dataBuffer.end() - 1)->descriptors, matches,
                       descriptorType_, matcherType, selectorType);

      // store matches in current data frame
      (dataBuffer.end() - 1)->kptMatches = matches;

      if (bDebug)
        cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;

      /* TRACK 3D OBJECT BOUNDING BOXES */

      // match list of 3D objects (vector<BoundingBox>) between
      // current and previous frame
      map<int, int> bbBestMatches;
      matchBoundingBoxes(
          matches, bbBestMatches, *(dataBuffer.end() - 2),
          *(dataBuffer.end() - 1)); // associate bounding boxes between current
                                    // and previous frame using keypoint matches

      // store matches in current data frame
      (dataBuffer.end() - 1)->bbMatches = bbBestMatches;

      if (bDebug)
        cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;

      /* COMPUTE TTC ON OBJECT IN FRONT */

      // loop over all BB match pairs
      for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin();
           it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1) {
        // cout << "BB Match: " << it1->first << ", " << it1->second << endl;
        // find bounding boxes associates with current match
        BoundingBox *prevBB, *currBB;
        for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin();
             it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2) {
          if (it1->second == it2->boxID) // check wether current match partner
                                         // corresponds to this BB
          {
            currBB = &(*it2);
          }
        }

        for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin();
             it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2) {
          if (it1->first == it2->boxID) // check wether current match partner
                                        // corresponds to this BB
          {
            prevBB = &(*it2);
          }
        }

        // compute TTC for current match
        if (currBB->lidarPoints.size() > 0 &&
            prevBB->lidarPoints.size() >
                0) // only compute TTC if we have Lidar points
        {
          // compute time-to-collision based on Lidar data
          double ttcLidar;
          computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints,
                          sensorFrameRate, ttcLidar);
          TTCsLidar.push_back(ttcLidar);

          // compute time-to-collision based on camera
          double ttcCamera;
          // assign enclosed keypoint matches to bounding box
          clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints,
                                   (dataBuffer.end() - 1)->keypoints,
                                   (dataBuffer.end() - 1)->kptMatches);
          computeTTCCamera((dataBuffer.end() - 2)->keypoints,
                           (dataBuffer.end() - 1)->keypoints,
                           currBB->kptMatches, sensorFrameRate, ttcCamera);
          TTCsCamera.push_back(ttcCamera);

          if (bVis || bSafe) {
            cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
            showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00,
                                R_rect_00, RT, &visImg);
            cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y),
                          cv::Point(currBB->roi.x + currBB->roi.width,
                                    currBB->roi.y + currBB->roi.height),
                          cv::Scalar(0, 255, 0), 2);

            char str[200];
            sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar,
                    ttcCamera);
            putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2,
                    cv::Scalar(0, 0, 255));

            ttcVideoWriter.write(visImg);
            if (bVis) {
              string windowName = "Final Results : TTC";
              cv::namedWindow(windowName, 4);
              cv::imshow(windowName, visImg);
              cout << "Press key to continue to next frame" << endl;
              cv::waitKey(0);
              cout << "TTC image size: " << visImg.size() << endl;
            }
          }

        } // eof TTC computation
      }   // eof loop over all BB matches
    }

    t_total = ((double)cv::getTickCount() - t_total) / cv::getTickFrequency();
    totalTimes.push_back(1000 * t_total / 1.0);

  } // eof loop over all images
  detObjVideoWriter.release();
  kptsVideoWriter.release();
  ttcVideoWriter.release();
  topviewVideoWriter.release();
}

void runObjectTracking(string detectorType, string descriptorType,
                       vector<vector<double>> &allDetectTimes,
                       vector<vector<double>> &allDescribeTimes,
                       vector<vector<double>> &allTotalTimes,
                       vector<vector<double>> &allTTCsCamera,
                       vector<double> &TTCsLidar, vector<string> &detDescTypes,
                       bool bVis, bool bDebug, bool bSafe) {
  // Save times needed for computation and TTCs
  vector<double> detectTimes;
  vector<double> describeTimes;
  vector<double> totalTimes;
  vector<double> TTCsCamera;

  object_tracking(detectorType, descriptorType, detectTimes, describeTimes,
                  totalTimes, TTCsCamera, TTCsLidar, bVis, bDebug, bSafe);
  allDetectTimes.push_back(detectTimes);
  allDescribeTimes.push_back(describeTimes);
  allTotalTimes.push_back(totalTimes);
  allTTCsCamera.push_back(TTCsCamera);
  detDescTypes.push_back(detectorType + "-" + descriptorType);
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[]) {
  /* INIT VARIABLES AND DATA STRUCTURES */
  bool bVis = false, bDebug = false,
       bSafe = true; // visualize results, print statements

  // Save times needed for computation and TTCs
  vector<vector<double>> allDetectTimes;
  vector<vector<double>> allDescribeTimes;
  vector<vector<double>> allTotalTimes;
  vector<vector<double>> allTTCsCamera;
  vector<double> TTCsLidar;
  vector<string> detDescTypes;

  // Detector and descriptor types
  // Detector types:
  // -> Gradient Based: HARRIS, SHITOMASI, SIFT
  // -> Binary: BRISK, ORB, AKAZE, FAST
  string detectorType;   // HARRIS, SHITOMASI, SIFT, BRISK, ORB, AKAZE, SIFT
  string descriptorType; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT

  // SIFT works only with SIFT
  detectorType = "SIFT";
  descriptorType = "SIFT";

  runObjectTracking(detectorType, descriptorType, allDetectTimes,
                    allDescribeTimes, allTotalTimes, allTTCsCamera, TTCsLidar,
                    detDescTypes, bVis, bDebug, bSafe);

  // AKAZE works only with AKAZE
  detectorType = "AKAZE";
  descriptorType = "AKAZE";
  runObjectTracking(detectorType, descriptorType, allDetectTimes,
                    allDescribeTimes, allTotalTimes, allTTCsCamera, TTCsLidar,
                    detDescTypes, bVis, bDebug, bSafe);

  // Try all other combinations of detector + descriptor
  vector<string> detectors{"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB"};
  vector<string> descriptors{"BRISK", "ORB", "BRIEF", "FREAK"};
  // vector<string> detectors{"SHITOMASI"};
  // vector<string> descriptors{"BRISK", "ORB"};

  for (string detectorType : detectors) {
    for (string descriptorType : descriptors) {
      runObjectTracking(detectorType, descriptorType, allDetectTimes,
                        allDescribeTimes, allTotalTimes, allTTCsCamera,
                        TTCsLidar, detDescTypes, bVis, bDebug, bSafe);
    }
  }
  cout << "Size of allTTCsCamera: " << allTTCsCamera.size() << endl;

  // Save stats
  ofstream ttc_lidar("../dat/stats/TTC_Lidar.txt");
  if (ttc_lidar.is_open()) {
    for (auto ttc : TTCsLidar) {
      ttc_lidar << ttc << "\n";
    }
    ttc_lidar.close();
  } else {
    cout << "Unable to open file";
  }

  ofstream ttc_camera("../dat/stats/TTC_Camera.txt");
  if (ttc_camera.is_open()) {
    for (auto type : detDescTypes) {
      ttc_camera << type << ", ";
    }
    ttc_camera << "\n";
    for (int i = 0; i < allTTCsCamera[0].size(); i++) {
      for (auto ttcs : allTTCsCamera) {
        ttc_camera << ttcs[i] << ", ";
      }
      ttc_camera << "\n";
    }
    ttc_camera.close();
  } else {
    cout << "Unable to open file";
  }

  ofstream ss1("../dat/stats/detectTimes.txt");
  if (ss1.is_open()) {
    for (auto type : detDescTypes) {
      ss1 << type << ", ";
    }
    ss1 << "\n";
    for (int i = 0; i < allDetectTimes[0].size(); i++) {
      for (auto stats : allDetectTimes) {
        ss1 << stats[i] << ", ";
      }
      ss1 << "\n";
    }
    ss1.close();
  } else {
    cout << "Unable to open file";
  }

  ofstream ss2("../dat/stats/describeTimes.txt");
  if (ss2.is_open()) {
    for (auto type : detDescTypes) {
      ss2 << type << ", ";
    }
    ss2 << "\n";
    for (int i = 0; i < allDescribeTimes[0].size(); i++) {
      for (auto stats : allDescribeTimes) {
        ss2 << stats[i] << ", ";
      }
      ss2 << "\n";
    }
    ss2.close();
  } else {
    cout << "Unable to open file";
  }

  ofstream ss3("../dat/stats/totalTimes.txt");
  if (ss3.is_open()) {
    for (auto type : detDescTypes) {
      ss3 << type << ", ";
    }
    ss3 << "\n";
    for (int i = 0; i < allTotalTimes[0].size(); i++) {
      for (auto stats : allTotalTimes) {
        ss3 << stats[i] << ", ";
      }
      ss3 << "\n";
    }
    ss3.close();
  } else {
    cout << "Unable to open file";
  }

  cout << "detectTimes: ";
  for (auto time : allDetectTimes[0]) {
    cout << time << ", ";
  }
  cout << endl;

  cout << "describeTimes: ";
  for (auto time : allDescribeTimes[0]) {
    cout << time << ", ";
  }
  cout << endl;

  cout << "totalTimes: ";
  for (auto time : allTotalTimes[0]) {
    cout << time << ", ";
  }
  cout << endl;

  return 0;
}
