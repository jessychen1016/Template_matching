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

for filename in glob.glob(inputdir+"227_contrast.jpg"):

	image = cv.imread(filename)

	#Gaussian Blur
	img_Gaussian = cv.GaussianBlur(image,(5,5),0,0)
	img_Gray = cv.cvtColor(img_Gaussian, cv.COLOR_BGR2GRAY)
	cv.imshow('Gaussian', img_Gray)
	cv.waitKey(2000)

	#Canny Feature 
	lthrehlod = 50
	hthrehlod = 150
	img_Canny = cv.Canny(img_Gray,lthrehlod,hthrehlod)
	cv.imshow('lines', img_Canny)
	cv.waitKey(10000)	

	

	# # Hough Transform
	# rho = 6
	# theta = np.pi/180
	# threhold =169
	# minlength = 40
	# maxlengthgap = 25
	# lines = cv.HoughLinesP(img_Canny,rho,theta,threhold,np.array([]),minlength,maxlengthgap)
	# linecolor =[0,255,255]
	# linewidth = 4

	# img_with_lines = img_Canny
	# for line in lines:
	# 	for x1,y1,x2,y2 in line:
	# 		cv.line(img_with_lines,(x1,y1),(x2,y2),linecolor,linewidth)




	picturename = os.path.basename(filename)
	print("picturename", picturename )
	cv.imwrite(outputdir+"canny_"+picturename, img_Canny, [cv.IMWRITE_JPEG_QUALITY, 100])
	print("Picture:", filename, "done")
	# cv.imshow('contrast', imgGRAY)
	# cv.waitKey(1000)
