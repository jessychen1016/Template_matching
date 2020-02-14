from __future__ import print_function
from builtins import input
import cv2 as cv
import numpy as np
import argparse
import glob
import os

# Read image given by user
parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! ')
parser.add_argument('--input', help='Path to input image.', default='./test/')
parser.add_argument('--output', help='Path to input image.', default='./contrast_convertBW_image/')
args = parser.parse_args()

inputdir = args.input
outputdir = args.output

for filename in glob.glob(inputdir+"277.jpg"):

	image = cv.imread(filename)
	
	for y in range(image.shape[0]):
	    for x in range(image.shape[1]):
	    	# if(image[y,x][0] == 255):
	    	# 	image[y,x][0],image[y,x][1],image[y,x][2] = 0,0,0
	    	if(image[y,x][0]>=60):
	    		image[y,x][0] = min(255, image[y,x][0]+60)
	    		image[y,x][1]=min(255,image[y,x][0]+3)
	    		image[y,x][2]=0



	picturename = os.path.basename(filename)
	print("picturename", picturename )
	cv.imwrite(outputdir+picturename, image, [cv.IMWRITE_JPEG_QUALITY, 100])
	print("Picture:", filename, "done")
	cv.imshow('modified', image)
	cv.waitKey(10000)
