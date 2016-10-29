#!/usr/bin/env python3.5
"""
importing some useful packages
"""

import argparse
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def load_image(image_path):
    """ pass for now """
    #return (mpimg.imread(image_path)*255).astype('uint8')
    return mpimg.imread(image_path)


def save_plot(dir_name, name, img):
    """ save plot to current directory """
    cv2.imwrite(join(dir_name, name), img)


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_noise(img, kernel_size):
    """ Applies a Gaussian Noise kernel """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    """ Applies the Canny transform """
    return cv2.Canny(img, low_threshold, high_threshold)


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
        ignore_mask_color = (255,)

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255,0,0], thickness=4):
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
    right_slope = []
    left_slope = []

    left_lines = []
    right_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            m = ((y2-y1)/(x2-x1)) # slope
            if m >= 0.5:
                right_slope.append(m)
                right_lines.append((x1,y1))
            elif m <= -.5:
                left_slope.append(m)
                left_lines.append((x2,y2))
    
    # average left and right slopes
    right_slope = sorted(right_slope)[int(len(right_slope)/2)]
    left_slope = sorted(left_slope)[int(len(left_slope)/2)]

    min_left_y1 = min([line[1] for line in left_lines])
    min_left_x1 = min([line[0] for line in left_lines if line[1] == min_left_y1])

    min_right_y1 = min([line[1] for line in right_lines])
    min_right_x1 = min([line[0] for line in right_lines if line[1] == min_right_y1])

    # x2 = ((y2-y1)/m) + x1 where y2 = max height
    # first we pick a start point on the horizon
    
    start_left_y = start_right_y = 325 # point on horizon
    start_right_x = int((start_right_y-min_right_y1)/right_slope) + min_right_x1
    start_left_x = int((start_right_y-min_left_y1)/left_slope) + min_left_x1

    # next we exten to the car
    end_left_x = int((img.shape[1]-min_left_y1)/left_slope) + min_left_x1
    end_right_x = int((img.shape[1]-min_right_y1)/right_slope) + min_right_x1
    
    cv2.line(img, (start_left_x, start_left_y), (end_left_x, img.shape[1]), color, thickness)
    cv2.line(img, (start_right_x, start_right_y), (end_right_x, img.shape[1]), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold,
                            np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    #for line in lines:
    #    for x1, y1, x2, y2 in line:
    #        cv2.line(line_img, (x1, y1), (x2, y2), [255,0,0], 2)
    return line_img


def weighted_img(img, initial_img, alpha=0.7, beta=0.5, upsilon=0.3):
    """
    Python 3 has support for cool math symbols.
    `img` is the output of the hough_lines(), An image with lines drawn on it
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + upsilon 
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, upsilon)


def parse_args():
    """ pass me """
    parser = argparse.ArgumentParser()
    parser.add_argument('test_on',
                        choices=('images', 'videos',),
                        help='Is this an image or a video test?')
    parser.add_argument('dir_path',
                        help='Path to directory containing images / videos')
    return parser.parse_args()

def get_files(dir_path):
    """ ok """
    return [f for f in listdir(dir_path) if isfile(join(dir_path, f))]



def process_img(name, img):
    """
    # 1: grayscale the image
    # 2: define edges
    # 3: Hough transform
    # 4: Apply ROI
    """
    dir_name = "output_images"
    initial_image = np.copy(img)
    
    gray_img = grayscale(img)
    blur_gray = gaussian_noise(gray_img, 3)
    edges = canny(blur_gray, 50, 150)
    #save_plot("output_images", "canny_"+name, edges)
    
    imshape = img.shape
    vertices = np.array([[(110,imshape[0]),(410, 320),(480, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    #save_plot(dir_name, "roi_"+name, masked_edges)
    
    lines = hough_lines(masked_edges, 1, np.pi/180, 15, 40, 30)
    #save_plot(dir_name, "line_"+name, lines)
    #save_plot(dir_name, "hough_"+name, lines)
    
    #lines = cv2.cvtColor(lines, cv2.COLOR_GRAY2RGB)
    zeros = np.zeros_like(lines)
    lines = np.dstack((lines, zeros, zeros))
    final_img = weighted_img(lines, initial_image)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    save_plot("output_images", "final_"+name, final_img)

def main():
    """ pass """
    args = parse_args()
    if args.test_on == 'images':
        images = get_files(args.dir_path)
        for name in images:
            if name.startswith("."): continue
            print("processing", name)
            img = load_image('{}/{}'.format(args.dir_path, name))
            process_img(name, img)

if __name__ == '__main__':
    main()