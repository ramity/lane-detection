import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import sys


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color = [52, 152, 219], thickness = 2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    # given a set of lines, we must find the average of a + and - slope lines

    positiveLines = 0
    negativeLines = 0

    positiveSlopeSum = 0
    negativeSlopeSum = 0

    positiveX1Sum = 0
    negativeX1Sum = 0

    positiveY1Sum = 0
    negativeY1Sum = 0

    positiveX2Sum = 0
    negativeX2Sum = 0

    positiveY2Sum = 0
    negativeY2Sum = 0

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = ((y2 - y1)/(x2 - x1))

                if(slope > 0):
                    # positive (ie left lane)
                    positiveLines += 1
                    positiveSlopeSum += slope
                    positiveX1Sum += x1
                    positiveY1Sum += y1
                    positiveX2Sum += x2
                    positiveY2Sum += y2
                else:
                    negativeLines += 1
                    negativeSlopeSum += slope
                    negativeX1Sum += x1
                    negativeY1Sum += y1
                    negativeX2Sum += x2
                    negativeY2Sum += y2

        if(positiveLines):
            # left lane

            positiveSlopeAverage = positiveSlopeSum / positiveLines
            positiveX1Average = positiveX1Sum / positiveLines
            positiveX2Average = positiveX2Sum / positiveLines
            positiveY1Average = positiveY1Sum / positiveLines
            positiveY2Average = positiveY2Sum / positiveLines

            cv2.putText(img, "left lane slope: %.4f" % positiveSlopeAverage, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 000, 255), 2)
            cv2.putText(img, "x1: %.4f" % positiveX1Average, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 000, 255), 2)
            cv2.putText(img, "y1: %.4f" % positiveY1Average, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 000, 255), 2)
            cv2.putText(img, "x2: %.4f" % positiveX2Average, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 000, 255), 2)
            cv2.putText(img, "y2: %.4f" % positiveY2Average, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 000, 255), 2)

            cv2.line(img, (int(positiveX1Average), int(positiveY1Average)), (int(positiveX2Average), int(positiveY2Average)), color, thickness + 5)

        if(negativeLines):
            # right lane

            negativeSlopeAverage = negativeSlopeSum / negativeLines
            negativeX1Average = negativeX1Sum / negativeLines
            negativeX2Average = negativeX2Sum / negativeLines
            negativeY1Average = negativeY1Sum / negativeLines
            negativeY2Average = negativeY2Sum / negativeLines

            cv2.putText(img, "right lane slope: %.4f" % negativeSlopeAverage, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 000, 255), 2)
            cv2.putText(img, "x1: %.4f" % negativeX1Average, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 000, 255), 2)
            cv2.putText(img, "y1: %.4f" % negativeY1Average, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 000, 255), 2)
            cv2.putText(img, "x2: %.4f" % negativeX2Average, (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 000, 255), 2)
            cv2.putText(img, "y2: %.4f" % negativeY2Average, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 000, 255), 2)

            cv2.line(img, (int(negativeX1Average), int(negativeY1Average)), (int(negativeX2Average), int(negativeY2Average)), color, thickness + 5)

        if(positiveLines and negativeLines):

            # right lane
            negativeSlopeAverage = negativeSlopeSum / negativeLines

            # left lane
            positiveSlopeAverage = positiveSlopeSum / positiveLines

            laneBias = negativeSlopeAverage + positiveSlopeAverage

            steeringColor = (255, 000, 000)

            if(laneBias > 0.2):
                steeringColor = (255, 255, 000)
                cv2.putText(img, "Suggested correction: VEER LEFT", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 000, 255), 2)
            elif(laneBias < -0.2):
                steeringColor = (255, 255, 000)
                cv2.putText(img, "Suggested correction: VEER RIGHT", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 000, 255), 2)
            else:
                steeringColor = (000, 255, 000)
                cv2.putText(img, "Suggested correction: CONT STRAIGHT", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 000, 255), 2)

            cv2.line(img, (10, 180), (210, 180), (255, 255, 255), 2)

            if(not np.isinf(laneBias)):
                scalingFactor = 100 / (abs(negativeSlopeAverage) + abs(positiveSlopeAverage))

                drivingXPosition = int(laneBias * scalingFactor) + 100

                cv2.circle(img, ((10 + drivingXPosition), 180), 8, steeringColor, 5)
            else:
                cv2.circle(img, ((10 + 100), 180), 8, steeringColor, 5)

            if(negativeSlopeAverage < -0.95 or positiveSlopeAverage > 0.95 or negativeSlopeAverage > 0.05 or positiveSlopeAverage < -0.05):
                cv2.putText(img, "DEPARTING LANE", (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 000, 000), 2)

        #saving just in case - previous implementation
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), [255, 0, 0], thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength = min_line_len, maxLineGap = max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α = 0.8, β = 1., λ = 0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_frame(image):
    global first_frame

    gray_image = grayscale(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #hsv = [hue, saturation, value]
    #more accurate range for yellow since it is not strictly black, white, r, g, or b

    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype = "uint8")

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 205, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

    kernel_size = 5
    gauss_gray = gaussian_blur(mask_yw_image, kernel_size)

    #same as quiz values
    low_threshold = 50
    high_threshold = 150
    canny_edges = canny(gauss_gray, low_threshold, high_threshold)

    #lewis updates:
    # NW    NE
    # SW    SE
    rows, cols = image.shape[:2]

    NWx = cols * 0.34
    NWy = rows * 0.43
    NEx = cols * 0.50
    NEy = rows * 0.45

    SWx = cols * 0.05
    SWy = rows * 0.67
    SEx = cols * 0.95
    SEy = rows * 0.75

    imshape = image.shape
    lowerLeft   = [SWx, SWy]
    lowerRight  = [SEx, SEy]
    topLeft     = [NWx, NWy]
    topRight    = [NEx, NEy]

    vertices = [np.array([lowerLeft, topLeft, topRight, lowerRight], dtype = np.int32)]
    roi_image = region_of_interest(canny_edges, vertices)

    #rho and theta are the distance and angular resolution of the grid in Hough space
    #same values as quiz
    rho = 1
    theta = np.pi / 180
    #threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 27
    min_line_len = 40
    max_line_gap = 130

    line_image = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)
    result = weighted_img(line_image, image, α = 0.8, β = 1., λ = 0.)
    return result

first_frame = 1
white_output = 'bot_lane_change_complete.mp4'
clip1 = VideoFileClip("bot_lane_change.mp4")
white_clip = clip1.fl_image(process_frame) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio = False)
