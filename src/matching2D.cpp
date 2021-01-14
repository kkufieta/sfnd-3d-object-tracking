#include "matching2D.hpp"
#include <numeric>

using namespace std;

bool debug = false;
// Find best matches for keypoints in two camera images based on several
// matching methods
float matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                       std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource,
                       cv::Mat &descRef, std::vector<cv::DMatch> &matches,
                       std::string descriptorType, std::string matcherType,
                       std::string selectorType) {
  // configure matcher
  bool crossCheck = false;
  cv::Ptr<cv::DescriptorMatcher> matcher;

  if (matcherType.compare("MAT_BF") == 0) {
    int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING
                                                             : cv::NORM_L2;
    matcher = cv::BFMatcher::create(normType, crossCheck);
    if (debug)
      cout << "BF matching";
  } else if (matcherType.compare("MAT_FLANN") == 0) {
    if (descSource.type() != CV_32F) { // OpenCV bug workaround : convert binary
                                       // descriptors to floating point due to a
                                       // bug in current OpenCV implementation
      descSource.convertTo(descSource, CV_32F);
      descRef.convertTo(descRef, CV_32F);
    }
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    if (debug)
      cout << "FLANN matching";
  }

  double t;
  // perform matching task
  if (selectorType.compare("SEL_NN") == 0) { // nearest neighbor (best match)
    t = (double)cv::getTickCount();
    matcher->match(
        descSource, descRef,
        matches); // Finds the best match for each descriptor in desc1
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    if (debug) {
      cout << " (NN) with n=" << matches.size() << " matches in "
           << 1000 * t / 1.0 << " ms" << endl;
    }
  } else if (selectorType.compare("SEL_KNN") ==
             0) { // k nearest neighbors (k=2)

    // Implement k-nearest-neighbor matching
    vector<vector<cv::DMatch>> knnMatches;
    int k = 2;
    t = (double)cv::getTickCount();
    matcher->knnMatch(descSource, descRef, knnMatches,
                      2); // finds the 2 best matches
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    if (debug) {
      cout << " (kNN) with k=2 and n=" << knnMatches.size() << " matches in "
           << 1000 * t / 1.0 << " ms" << endl;
    }

    // Filter matches using descriptor distance ratio test
    float ratio;
    int numTotalMatches = knnMatches.size();
    int numDiscardedMatches = 0;
    double minDistanceRatio = 0.8;
    for (vector<cv::DMatch> kMatch : knnMatches) {
      ratio = kMatch[0].distance / kMatch[1].distance;
      if (ratio <= minDistanceRatio) {
        matches.push_back(kMatch[0]);
      } else {
        numDiscardedMatches += 1;
      }
    }
    if (debug) {
      cout << "Number discarded matches: " << numDiscardedMatches << endl;
      cout << "Percentage discarded matches: "
           << numDiscardedMatches / (float)numTotalMatches << endl;
    }
  }
  return 1000 * t / 1.0;
}

// Use one of several types of state-of-art descriptors to uniquely identify
// keypoints
float descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                    cv::Mat &descriptors, string descriptorType) {
  // select appropriate descriptor
  cv::Ptr<cv::DescriptorExtractor> extractor;
  if (descriptorType.compare("BRISK") == 0) {

    int threshold = 30;        // FAST/AGAST detection threshold score.
    int octaves = 3;           // detection octaves (use 0 to do single scale)
    float patternScale = 1.0f; // apply this scale to the pattern used for
                               // sampling the neighbourhood of a keypoint.

    extractor = cv::BRISK::create(threshold, octaves, patternScale);
  } else if (descriptorType.compare("SIFT") == 0) {
    extractor = cv::SIFT::create();
  } else if (descriptorType.compare("ORB") == 0) {
    extractor = cv::ORB::create();
  } else if (descriptorType.compare("AKAZE") ==
             0) { // works only with AKAZE keypoints
    extractor = cv::AKAZE::create();
  } else if (descriptorType.compare("BRIEF") == 0) {
    extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
  } else if (descriptorType.compare("FREAK") == 0) {
    extractor = cv::xfeatures2d::FREAK::create();
  }

  // perform feature description
  double t = (double)cv::getTickCount();
  extractor->compute(img, keypoints, descriptors);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  if (debug) {
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0
         << " ms" << endl;
  }
  return 1000 * t / 1.0;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
float detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                            bool bVis) {
  // compute detector parameters based on image size
  int blockSize = 4; //  size of an average block for computing a derivative
                     //  covariation matrix over each pixel neighborhood
  double maxOverlap = 0.0; // max. permissible overlap between two features in %
  double minDistance = (1.0 - maxOverlap) * blockSize;
  int maxCorners =
      img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

  double qualityLevel = 0.01; // minimal accepted quality of image corners
  double k = 0.04;

  // Apply corner detection
  double t = (double)cv::getTickCount();
  vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance,
                          cv::Mat(), blockSize, false, k);

  // add corners to result vector
  for (auto it = corners.begin(); it != corners.end(); ++it) {

    cv::KeyPoint newKeyPoint;
    newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
    newKeyPoint.size = blockSize;
    keypoints.push_back(newKeyPoint);
  }
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  if (debug) {
    cout << "Shi-Tomasi corner detection with n=" << keypoints.size()
         << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  }

  // visualize results
  if (bVis) {
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "Shi-Tomasi Corner Detector Results";
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
  }
  return 1000 * t / 1.0;
}

// Detect keypoints in image using the traditional Harris detector
float detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                         bool bVis) {
  // compute detector parameteres based on image size
  int blockSize = 2;
  int apertureSize = 3; // Aperture parameter for Sobel parameter
  int minResponse =
      100; // minimum value for a corner in the 8bit scaled response matrix
  double k = 0.04; // Harris parameter (see equation for details)

  // Detect Harris corners and normalize output
  cv::Mat dst, dst_norm, dst_norm_scaled;
  dst = cv::Mat::zeros(img.size(), CV_32FC1);
  double t = (double)cv::getTickCount();
  cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  cv::convertScaleAbs(dst_norm, dst_norm_scaled);

  // Locate local maxima in the Harris response matrix
  // and perform a non-maximum suppression (NMS) in a local neighborhood around
  // each maximum. The resulting coordinates are stored in a list of
  // keypoints of the type `vector<cv::KeyPoint>`.
  double maxOverlap = 0.0; // max. permissible overlap
  for (size_t r = 0; r < dst_norm_scaled.rows; r++) {
    for (size_t c = 0; c < dst_norm_scaled.cols; c++) {
      int response = (int)dst_norm.at<float>(r, c);
      if (response > minResponse) {
        cv::KeyPoint keypoint;
        keypoint.pt = cv::Point2f(c, r);
        keypoint.size = 2 * apertureSize;
        keypoint.response = response;

        // Perform non-maximum suppression (NMS) in local neighborhood around
        // new keypoint
        bool bOverlap = false;
        for (auto it = keypoints.begin(); it != keypoints.end(); it++) {
          float kptOverlap = cv::KeyPoint::overlap(keypoint, *it);
          if (kptOverlap > maxOverlap) {
            bOverlap = true;
            if (keypoint.response > (*it).response) {
              *it = keypoint; // replace old keypoint with new one
              // break;
            }
          }
        }
        // Only add new keypoint if overlap has not been found in NMS
        if (!bOverlap) {
          keypoints.push_back(keypoint);
        }
      }
    }
  }

  if (debug) {
    cout << "Harris corner detection with n=" << keypoints.size()
         << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  }
  if (bVis) {
    // visualize results
    cv::Mat vis_image = dst_norm_scaled.clone();
    cv::drawKeypoints(dst_norm_scaled, keypoints, vis_image);
    string windowName = "Harris Corner Detector with NMS Results";
    cv::namedWindow(windowName, 6);
    cv::imshow(windowName, vis_image);
    cv::waitKey(0);
  }
  return 1000 * t / 1.0;
}

// SIFT, FAST, BRISK, ORB, AKAZE
float detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                         std::string detectorType, bool bVis) {
  // Detect keypoints
  double t = (double)cv::getTickCount();
  if (detectorType.compare("SIFT") == 0) {
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    detector->detect(img, keypoints);
  } else if (detectorType.compare("FAST") == 0) {
    cv::Ptr<cv::FastFeatureDetector> detector =
        cv::FastFeatureDetector::create();
    detector->detect(img, keypoints);
  } else if (detectorType.compare("BRISK") == 0) {
    cv::Ptr<cv::BRISK> detector = cv::BRISK::create();
    detector->detect(img, keypoints);
  } else if (detectorType.compare("ORB") == 0) {
    cv::Ptr<cv::ORB> detector = cv::ORB::create(5000);
    detector->detect(img, keypoints);
  } else if (detectorType.compare("AKAZE") == 0) {
    cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
    detector->detect(img, keypoints);
  }
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  if (debug) {
    cout << detectorType << " corner detection with n=" << keypoints.size()
         << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  }
  return 1000 * t / 1.0;
}