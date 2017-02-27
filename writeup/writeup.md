#**Finding Lane Lines on the Road** 

##Writeup


**Finding Lane Lines on the Road**

The goal of this project was to build a pipeline that process still images or video and detects lane lines on a road.
The objective have been reached to a point, but there are several shortcomings listed at the end, along with suggestions for improvements.



[//]: # (Image References)

[image1]: ./step1_grayscale.png "Grayscale"
[image2]: ./step2_blur.png "Blur"
[image3]: ./step3_canny.png "Canny edge detection"
[image4]: ./step4_mask.png "Mask"
[image5]: ./step5_hough.png "Hough transform to find straight lines"
[image6]: ./average_line.png "average lines found by hough algorithm to draw a single line"

---

### Reflection

###1. Pipeline

The pipeline consist of 5 steps. 

* 1: convert to grayscale
The image must be in single color, 8 bits/pixel for following process, so we convert to grayscale using OpenCV cvtColor function.

![alt text][image1]

* 2: blur image
There may be small hard edges on the image that we are not interested in, result from the digitalization or reflections or other factors.
We strive to eliminate these by blurring the image using gaussian blur.

![alt text][image2]

* 3: detect edges
Now that the image is blurred, we apply the Canny algorithm to find edges

![alt text][image3]

* 4: isolate ROI
We are not interested on the whole image but only on a region: the region of interest or ROI. We define a ROI by a polygon to include only the lane, and apply masking to remove the contents of the image outside of the ROI

![alt text][image4]

* 5: find lines
Now we have only the edges on the region of interest, so we apply the Hough transform to find lines on it.
To visualize, we superimpose the lines found over the original image:

![alt text][image5]


Finally, the draw_lines() function has to be modified to extract a single line for each lane. Each line found by Hough function has the form 
y = m * x + b. We calculate m and b for each line and separate them in left and write lane using the slope: 

* if m < certain inclination, the line is too horizontal to belong to a lane and is ignored.
* if m is infinite, the line is vertical and it is ignored.
* if m < 0, the line is part of the left lane line
* if m > 0, the line is part of the right lane line

Note that the y coordinate runs from top to bottom, that's why the left/right signs are inverted.

There may be frames in a video that lack one or more lane lines due to light/contrast conditions; the code tackles this by keeping the last found averages and using those.



###2. Shortcomings
There are several potential shortcomings on this pipeline:
* the pipeline is not applicable to any image/video, as it is dependant on several parameters that have to be adjusted for each one.
* the region of interest have to be adjusted for each image based on image size, camera position and orientation, etc. 
* the contrast between the lines and the background has to be enough to detect the edge. This will vary with lighting conditions, road marks, even road construction materials
* there are wide variations on the lane lines length. There are solid lines, dots, dashes, and every combination. This makes the finding of lines a challenge.
* when applying this pipeline to a series of images (a video), all this problems can appear from one frame to the next. This would require an 
 automatic adjustment of parameters from frame to frame.

I applied the pipeline to a video taken during a trip with my family, and it clearly shows that is it very difficult to keep the car steady enough on the center of the lane to 
use the same ROI from frame to frame, along with other problems like other markings on the road that have straight lines and are detected by Hough transform.
And of course, the camera should be stable! I filmed this with my phone on my hand, so the center changes erratically and there are 'rogue' lines on several frames.



###3. Possible improvements
A possible improvement would be to somehow process the image to eliminate the effect of lighting variations before detecting edges.
This will allow the detection of lanes in different weather conditions or with shadows cast by trees for example.

Another improvement could be to use a more strict region of interest that 'wraps' around the lane lines so to remove any interior markings.
In my own video can be seen the 'fog marks' which are small arrowheads inside the lanes; these confound the lane detector pipeline because they are made of straight lines at not so small angles.
The car's own shadow confounds the pipeline sometimes too.

