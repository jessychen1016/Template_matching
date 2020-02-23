import cv2 as cv
import numpy as np
import argparse
import glob
import os
import math
import time

parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! ')
parser.add_argument('--input', help='Path to input image.', default='./')
parser.add_argument('--output', help='Path to input image.', default='./hole_filling/')
parser.add_argument('--cuda', help='whether to use CUDA.', default=0)
args = parser.parse_args()

inputdir = args.input
outputdir = args.output

for filename in glob.glob(inputdir+"ortho_no_car.jpeg"):

	image = cv.imread(filename)
	i=0
	for y in range(0, image.shape[0], 10):
	    for x in range(0, image.shape[1], 10):
	    	if(image.shape[0]-666-y>=0 and image.shape[1]-666-x>=0):
		    	crop_image = image[y:666+y,x:666+x]
		    	picturename = os.path.basename(filename)
		    	# print("picturename", picturename )
		    	cv.imwrite(outputdir+str(i)+'_'+picturename, crop_image, [cv.IMWRITE_JPEG_QUALITY, 100])
		    	# print("Picture:", filename, "done")
		    	# cv.imshow('modified', crop_image)
		    	# cv.waitKey(1000)
		    	i += 1