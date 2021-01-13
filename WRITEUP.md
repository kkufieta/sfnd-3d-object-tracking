# Track an object in 3D space
NOTES to self: 
* Include statement & supporting figures/images
* How did I address things?
* Where in the code did I address them?

## Match 3D Objects
**Task:** Implement the method `matchBoundingBoxes`, which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences. Each bounding box is assigned the match candidate with the highest number of occurrences.

**Implementation:**
`matchBoundingBoxes` is located in `camFusion.cpp` and looks like this:
```cpp
void matchBoundingBoxes(std::vector<cv::DMatch> &matches,
                        std::map<int, int> &bbBestMatches, DataFrame &prevFrame,
                        DataFrame &currFrame) {
  // match list of 3D objects (vector<BoundingBox>) between
  // current and previous frame
  // Loop through all matches and check in which box the respective keypoints
  // lie in. Save those bounding box id matches to bbMatches.
  std::multimap<int, int> bbMatches;
  int prevBoxID = -1;
  int currBoxID = -1;
  for (cv::DMatch match : matches) {
    prevBoxID = -1;
    currBoxID = -1;
    for (BoundingBox bbox : prevFrame.boundingBoxes) {
      if (bbox.roi.contains(prevFrame.keypoints[match.queryIdx].pt)) {
        prevBoxID = bbox.boxID;
      }
    }
    for (BoundingBox bbox : currFrame.boundingBoxes) {
      if (bbox.roi.contains(currFrame.keypoints[match.trainIdx].pt)) {
        currBoxID = bbox.boxID;
      }
    }
    if (prevBoxID >= 0 && currBoxID >= 0) {
      bbMatches.insert(std::pair<int, int>(prevBoxID, currBoxID));
    }
  }

  // Loop through bbMatches for every bounding box in the previous frame.
  // Save the frequency for the matched boxes.
  // Get the box in the current frame that has been matched the most to the
  // box in the previous frame.
  for (int prevBoxID = 0; prevBoxID < prevFrame.boundingBoxes.size(); prevBoxID++) {
    vector<int> countMatches(currFrame.boundingBoxes.size(), 0);
    std::pair<std::multimap<int, int>::iterator,
              std::multimap<int, int>::iterator>
        ret = bbMatches.equal_range(prevBoxID);
    for (auto it = ret.first; it != ret.second; it++) {
      countMatches[it->second] += 1;
    }
    int matchedBoxId =
        std::max_element(countMatches.begin(), countMatches.end()) -
        countMatches.begin();
    bbBestMatches[prevBoxID] = matchedBoxId;
  }
}
```

The function takes the matches for the previous and current frame, and loops through them. for every match, it loops through the bounding boxes in the previous and current frames. If a bounding box in both frames contains the match, add the pair of box ids that belong to them to the multimap `bbMatches`. 

After processing all matches and populating `bbMatches`, we loop through `bbMatches` for every bounding box in the previous frame. We get the matching bounding box IDs in the current frame with `bbMatches.equal_range(prevBoxID)`. We loop through them, and count how many times each bounding box in the current frame comes up. 

Eventually, we look up the highest number of keypoint occurences for the bounding boxes in the current frame, and pick the respective box as a match for the box in the previous frame. We save the pairs in `bbBestMatches`.


## Compute Lidar-based TTC
**Task:** Compute the time-to-collision (TTC) in seconds for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame. Deal with outlier Lidar points in a statistically robust way to avoid severe estimation errors.

**Implementation:** 
`computeTTCLidar` is located in `camFusion.cpp` and looks like this:
```cpp
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate,
                     double &TTC) {
  double dT = 1 / frameRate;
  // NAIVE IMPLEMENTATION: Pick the points with the smallest x-value:
  // double closestXPrev = lidarPointsPrev[0].x;
  // double closestXCurr = lidarPointsCurr[0].x;
  // for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); it++)
  // {
  //   closestXPrev = it->x < closestXPrev ? it->x : closestXPrev;
  // }
  // for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); it++)
  // {
  //   closestXCurr = it->x < closestXCurr ? it->x : closestXCurr;
  // }
  // TTC = closestXCurr * dT / (closestXPrev - closestXCurr);

  // MORE ROBUST IMPLEMENTATION: In order to filter out outliers, choose the
  // median of the smallest 20 x values
  auto compareX = [](LidarPoint lp1, LidarPoint lp2) {
    return (lp1.x < lp2.x);
  };
  std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), compareX);
  std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), compareX);
  int nPrev = lidarPointsPrev.size() < 20 ? lidarPointsPrev.size() : 20;
  int nCurr = lidarPointsCurr.size() < 20 ? lidarPointsCurr.size() : 20;
  double medianSmallestXPrev = (lidarPointsPrev[std::ceil(nPrev / 2. - 1)].x +
                                lidarPointsPrev[std::floor(nPrev / 2.)].x) /
                               2.0;
  double medianSmallestXCurr = (lidarPointsCurr[std::ceil(nCurr / 2. - 1)].x +
                                lidarPointsCurr[std::floor(nCurr / 2.)].x) /
                               2.0;
  TTC = medianSmallestXCurr * dT / (medianSmallestXPrev - medianSmallestXCurr);
}
```
The first implementation was a naive implementation (now commented out, but left in for reference), where we looked for a single point in both the previous and current frame that had the smallest x-value. This solution is prone to measurement errors, which can lead to severe estimation errors. 

The second solution is slightly more robust: We sort the lidar points based on the x-value, and pick 20 points (or fewer if not enough available) with the smallest x-values. The basis for calculating the TTC is the median of those 20 values.

The TTC is then calculated based on the assumption that the velocity is constant. 

Note: The TTC estimation can yield very high values when `(medianSmallestXPrev - medianSmallestXCurr)` are close to zero, which happens when the distance between the ego car and the car ahead doesn't change between successive images. It would probably be a good idea to add apply a smoothing function (low-pass filter) over the TTC.

## Associate Keypoint Correspondences with Bounding Boxes
**Task:** Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

Add the keypoint correspondences to the `kptMatches` property of the respective bounding boxes. Remove outlier matches based on the euclidean distance between them in relation to all the matches in the bounding box.

**Implementation:**
The code for this task is in the function `clusterKptMatchesWithROI`, is located in `camFusion.cpp` and looks like this:
```cpp
// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox,
                              std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr,
                              std::vector<cv::DMatch> &kptMatches) {
  for (cv::DMatch match : kptMatches) {
    if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt)) {
      boundingBox.keypoints.push_back(kptsCurr[match.trainIdx]);
      boundingBox.kptMatches.push_back(match);
    }
  }
  vector<double> euclideanDistances;
  for (cv::DMatch match : boundingBox.kptMatches) {
    double euclDist =
        cv::norm(kptsPrev[match.queryIdx].pt - kptsCurr[match.trainIdx].pt);
    euclideanDistances.push_back(euclDist);
  }
  int n = euclideanDistances.size();
  std::sort(euclideanDistances.begin(), euclideanDistances.end());
  double medianEuclideanDistance = (euclideanDistances[std::ceil(n / 2. - 1)] +
                                    euclideanDistances[std::floor(n / 2.)]) /
                                   2.0;
  vector<cv::KeyPoint> kpts;
  vector<cv::DMatch> matches;
  for (cv::DMatch match : boundingBox.kptMatches) {
    double euclDist =
        cv::norm(kptsPrev[match.queryIdx].pt - kptsCurr[match.trainIdx].pt);
    if (abs(euclDist - medianEuclideanDistance) <= 25) {
      matches.push_back(match);
      kpts.push_back(kptsCurr[match.trainIdx]);
    }
  }
  boundingBox.keypoints = kpts;
  boundingBox.kptMatches = matches;
}
```
Here we cluster the keypoint matches that belong to a specific bounding box, defined by `boundingBox`. 

We loop through all matches, and if the current bounding box contains the relevant keypoint, we add both the keypoint and the match to vectors that are attached to the bounding box. 

Next, in order to remove outliers, we loop over all the matches contained in the current box. We store the euclidean distance betwen the two keypoints in an array. Then, we calculate the median euclidean distance based on the values in the array. Given the median euclidean distance, we loop again over all the matches and save only those that don't deviate too much from it (here an devation of up to 25 is allowed).

## Compute Camera-based TTC
**Task:** Compute the time-to-collision in seconds for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame. Deal with outlier correspondences in a statistically robust way to avoid severe estimation errors.

**Implementation:** 
`computeTTCCamera` is located in `camFusion.cpp` and looks like this:
```cpp
// Compute time-to-collision (TTC) based on keypoint correspondences in
// successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev,
                      std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate,
                      double &TTC, cv::Mat *visImg) {
  // compute distance ratios between all matched keypoints
  vector<double> distRatios; // stores the distance ratios for all keypoints
                             // between curr. and prev. frame
  for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1;
       ++it1) { // outer keypoint loop

    // get current keypoint and its matched partner in the prev. frame
    cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
    cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

    for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end();
         ++it2) { // inner keypoint loop

      double minDist = 100.0; // min. required distance

      // get next keypoint and its matched partner in the prev. frame
      cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
      cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

      // compute distances and distance ratios
      double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
      double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

      if (distPrev > std::numeric_limits<double>::epsilon() &&
          distCurr >= minDist) { // avoid division by zero

        double distRatio = distCurr / distPrev;
        distRatios.push_back(distRatio);
      }
    } // eof inner loop over all matched kpts
  }   // eof outer loop over all matched kpts

  // only continue if list of distance ratios is not empty
  if (distRatios.size() == 0) {
    TTC = NAN;
    return;
  }

  // compute camera-based TTC from distance ratios
  double dT = 1 / frameRate;
  std::sort(distRatios.begin(), distRatios.end());
  int n = distRatios.size();
  double medianDistRatio =
      (distRatios[std::ceil(n / 2. - 1)] + distRatios[std::floor(n / 2.)]) /
      2.0;
  TTC = -dT / (1 - medianDistRatio);
}
```
To compute the TTC using the camera, we need to calculate the distance ratios between all matched keypoints. When points come closer, the distance to the car increases. When their distance becomes larger, the car is cloer up. Thus, the distance ratio can be used to calculate the TTC.

In order to calculate the distance ratio, we loop through all keypoint matches in a double nested loop. We compare the distances between keypoints from the previous and current frame. By dividing them, we get the distance ratio, which we save in an array.

From that array, we use the median distance ratio to calculate the TTC.

Note: The TTC is prone to have a very large value if the median distance ratio `medianDistRatio` is close to 1. This would happen when the size of the car ahead is not changing much from picture to picture, indicating that the distance between the cars is not changing. This is a similar problem as we have with computing the TTC using the lidar. Again, one way to improve this would be to use a smoothing function and to consider the TTC from a couple of previous data frames.

## Performance Evaluation
### Analysis 1
**Task:** Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

Identify and describe several examples (2-3) in detail. The assertion that the TTC is off has been based on manually estimating the distance to the rear of the preceding vehicle from a top view perspective of the Lidar points.

### Analysis 2
**Task:** Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

Compare all detector / descriptor combinations implemented in previous chapters with regard to the TTC estimate on a frame-by-frame basis. To facilitate comparison, a spreadsheet and graph should be used to represent the different TTCs.