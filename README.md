# carND-LaneLines
Project 1 for Self-driving cars nanodegree

This project implements a pipeline to process an image or video of a car driving on a road with lane lines.
The code will identify the lane lines and draw red lines over the original image/video.

[//]: # (Image References)

[image1]: ./writeup/step1_grayscale.png "Grayscale"
[image2]: ./writeup/step2_blur.png "Blur"
[image3]: ./writeup/step3_canny.png "Canny edge detection"
[image4]: ./writeup/step4_mask.png "Mask"
[image5]: ./writeup/step5_hough.png "Hough transform to find straight lines"
[image6]: ./writeup/average_line.png "average lines found by hough algorithm to draw a single line"

The pipeline consist of 5 steps: 

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

