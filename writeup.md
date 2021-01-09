## Writeup 

Please note that the final submission file is **Advanced Lane Finding.ipynb** in the main directory.
There is an additional IPython notebook titled **images_for_writeup.ipynb** which was used to
generate the images in this writeup. The final output video is called **project_output.mp4** and all the images in the writeup 
can be found in the directory `./output_images`
---
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1a]: ./output_images/chess_undistorted.png "Undistorted chessboard"
[image1b]: ./output_images/test3_undistorted.png "Undistorted test"
[image2]: ./output_images/thresholded_output.png "Binary Example"
[image3]: ./output_images/warped_output.png "Road transformed"
[image4]: ./output_images/sliding_window.png "Sliding Window"
[image5]: ./output_images/search_from_prior.png "Search from Prior"
[image6]: ./output_images/out_image.png "Output"
[video1]: ./project_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Below is the detailed write-up addressing different points in the rubric. Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

The code for this step is contained in the **second code cell** of the IPython notebook located in `./Advanced Lane Finding.ipynb`.  
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  This function returns is implemented as calc_calibration_matrix(), and returns the calibration matrix `cal_mtx`, and the distortion matrix `dist`. I use `dist` with `cv2.undistort()` function for distortion correction. The distortion correction applied on the chessboard image `./camera_cal/calibration1.jpg` is shown below:

![Chessboard Undistorted][image1a]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
Similar to the chessboard image, I applied the distortion correction on one of the test images - `./test_images/test3.jpg`:

![Test image Undistorted][image1b]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding function iscontained in the **third code cell** of the IPython notebook located in `./Advanced Lane Finding.ipynb`). The threshold values for the different colors (white, and yellow lanes), and the gradient were chosen after several trials on the test images. Here's an example of my output for this step.

![Binary Thresholded Image][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is contained in the fourth code cell of the IPython notebook located in `./Advanced Lane Finding.ipynb`.  The `perspective_transform()` function takes as inputs an image (`img`) and returns the warped image and the inverse transformation matrix which will be used later in the pipeline. The `perspective_transform()` function provides a birds-eye
view of the road lanes. The source points (`src`) and destination (`dst`) points are hard coded as per the `./test_images/straight_lines1.jpg` image. Since the camera remains unchanged, we can use the same matrix to transform all image frames.  I chose the hardcoded source and destination points in the following manner:

```python
img_width = img.shape[1]
img_height = img.shape[0]

# Compute src points for transform
s1 = [img_width//2 - 75, 450]  # Trapezoid edge is at 450
s2 = [img_width//2 + 75, 450]
s3 = [-100, img_height]
s4 = [img_width + 100, img_height]
src = np.float32([s1, s2, s3, s4])

# Compute dst points for transform
dst = np.float32([[100, 0], 
                  [img_width-100, 0], 
                  [100, img_height], 
                  [img_width-100, img_height]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 565, 450      | 100, 0        | 
| 715, 450      | 1180, 0       |
| -100, 720     | 100, 720      |
| 1380, 720     | 1180, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Road lanes after Perspective Transform][image3]

We can see that the straight lines image appear as parallel lanes after applying perspective transform.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The crux of the pipeline is to identify the lane lines. This is implemented in **code cells five through ten** in the IPython notebook `./Advanced Lane Finding.ipynb`. 
I first create a class Line to store relevant attributes of the lane lines as suggested under
_Tips and tricks for the project_. I then instantiate two objects namely, `left_lane` and `right_lane` to store the
attributes of the two lane lines respectively. Note that these objects are global. 

Initially, I set both `left_lane.detected` and `right_lane.detected` as `False`. From the pipeline, after thresholding and
warping the image, I call the function `find_lane_pixels()` (implemented in code cell ten).
This function will either identify the lane lines with the sliding window approach (if lane lines were not identified
in the previous frame, or if this is the first frame), or else will search around the lane lines detected in the previous frame.

The sliding window approach is implemented in the **eighth code cell**. It first calculates the base position of the two lane lines
using histogram of the binary thresholded image, and then uses windows of fixed size to search for nonzero pixels in the
rest of the frame. These nonzero pixels are appended to `left_lane_inds`, and `right_lane_inds`. If the number of nonzero
pixels detected are greater than a certain threshold, we then update the centre of the window to the average of the 
detected nonzero pixels. Note that separate windows are used for the left and right lanes and updated accordingly. I then
use a 2nd order polynomial to fit the nonzero pixels of the left lane and right lane using the `np.polyfit()` function.
The fitted coefficients are stored in `left_lane.current_fit`, and `right_lane.current_fit`. I also set `left_lane.detected` and
`right_lane.detected` as `True`. The sliding window approach for detecting lane lines is shown below. Note that the two 
detected lane lines are colored differently, and the green color indicates the windows.

![Sliding Window][image4]

The `search_from_prior()` function (implemented in **code cell nine**) is called when lane lines are detected in the previous frame.
Since it is computationally expensive to detect lane lines using the sliding window approach on every frame, we instead simply search
for nonzero pixels within a thresholded area of the previous fitted lines (as the lane lines will not change drastically 
in consecutive frames). I then use the nonzero pixels detected in the current frame to update the fit of the lane lines.
Here again, a 2nd order polynomial is used to fit the nonzero pixels as done in sliding window approach.

In case, sufficient nonzero pixels aren't detected in the current frame near the previously fitted lines,
then I set `left_lane.detected` and `right_lane.detected` as `False`, and use sliding window approach for the current frame.
The ouptut from `search_from_prior()` function is shown below. The yellow lines are the fitted lane lines, and the transparent
green area shows the search boundary for nonzero pixels based on the fit from the previous frame.

![Search from Prior][image5]

**Additional Note**: My previous submission would sometimes detect erroneous lane lines. To overcome that, I verify by ensuring that the radius of curvature of the two detected lane lines aren't too small, and I also measure the distance between the lanes, and make sure that it is within 3.7+/-0.3m, as the actual lane width is 3.7 meters. If any of these conditions fail, then I continue to use the fit co-efficients from the previous frame.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Measuring radius of curvature is contained in the **11th code cell** and measuring the position of the vehicle with respect to the lane centre is
implemented in the **12th code cell** in the IPython notebook `./Advanced Lane Finding.ipynb`.

Radius of curvature is given by the formula:
$$ R_{curve} = \frac{(1+(2Ay+B)^2)^{3/2}}{|2A|}$$
where A and B are coefficients from the 2nd order polynomial fit. Since we want to calculate the radius of curvature in meters, and not in 
pixels, I first convert the identified nonzero pixels in the current frame from pixels to meter (the conversion is explained in the pipeline)
and then obtain the fit coefficients in the meter space. I then use the above formula to
calculate the radius of curvature of the left and right lanes and return the average radius of curvature.

To measure distance between the vehicle centre and the lane centre, we assume that the camera is mounted on the centre of the vehicle, 
in which case the centre of the image is equivalent to the vehicle centre:
```python
img_centre = img.shape[1]/2
```

I find the x-coordinates of the fitted lines at the bottom of the image for the left and right lanes, and call them
`leftx`, and `rightx` respectively. I calculate the lane centre as the average of the 2 x-coordinates:
```python
lane_centre = (leftx + rightx)/2.0
```

and then return the difference between the image centre and the lane centre converted to meters.
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The complete pipeline is defined in **code cell 13** of the Ipython notebook `./Advanced Lane Finding.ipynb`. After performing all the steps listed above,
I use `cv2.fillPoly()` function to fill the lane in green, and warp the image back to the original using `Minv` from perpective
transform. I also add information such as the radius of curvature, and the vehicle offset using `cv2.putText`. If the radius of curvature
is too high, then I state that the road is straight. Here is an example of my result on a test image:

![Output][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Challenges Faced: 
- I took sometime to get the color and gradient thresholds working on all images. Even now, I can't say it is
completely robust. Since all the test images were mostly in daylight, I'm not sure how well these thresholds will
work under night conditions.
- It took me a while to implement search from prior fit lines and to enable switching between
sliding window approach to the search from prior fit lines. 
- I also realized that I had initially made an error while calculating from pix to meters and hence I was getting
some incorrect answers for the radius of curvature.

Suggestions for Improvement:
- I did not use all the attributes in the given Line class. A better approach would be to keep the average fit from the last n 
frames, and utilize those fit coefficients in case lane lines go undetected in a few frames.
- I can also store the difference between fit coefficients from consecutive frames and ensure that the difference is not 
too large to verify that the lanes are fitted correctly.
- More testing and tweaking to ensure that the thresholding works effectively in different scenarios.
- I can also compartmentalize my code for better readability, instead of using just one IPython Notebook.
