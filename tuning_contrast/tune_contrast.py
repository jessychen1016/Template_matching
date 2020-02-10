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

for filename in glob.glob(inputdir+"*.jpg"):

	image = cv.imread(filename)
	# if image is None:
	#     print('Could not open or find the image: ', args.input)
	#     exit(0)
	# new_image = np.zeros(image.shape, image.dtype)
	# alpha = 1.0 # Simple contrast control
	# beta = 0    # Simple brightness control
	# # Initialize values
	# print(' Basic Linear Transforms ')
	# print('-------------------------')
	# try:
	#     alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
	#     beta = int(input('* Enter the beta value [0-100]: '))
	# except ValueError:
	#     print('Error, not a number')
	# # # Do the operation new_image(i,j) = alpha*image(i,j) + beta
	# # # Instead of these 'for' loops we could have used simply:
	# # # new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
	# # # but we wanted to show you how to access the pixels :)
	# # for y in range(image.shape[0]):
	# #     for x in range(image.shape[1]):
	# #         for c in range(image.shape[2]):
	# #             new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
	# new_image=cv.convertScaleAbs(image, alpha=alpha, beta=beta)

	imghsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)


	# for row in imghsv[:,:,2]:
	# 	for pixel in row:
	# 		if pixel == 255:
	# 			pixel = 255
	# 		elif pixel < 60:
	# 			pixel=max(0.2*pixel, 0)
	# 		else:
	# 			pixel = pixel + 0.5*(255-pixel)

# for the ground pictures
	imghsv[:,:,2] = [[max(0.6*pixel, 0) if pixel < 60 else min(pixel+0.8*(255-pixel), 255) if pixel<155 else 0 for pixel in row] for row in imghsv[:,:,2]]
	
# # for the orthomosaic
# 	imghsv[:,:,2] = [[max(0.3*pixel, 0) if pixel < 100 else min(pixel+0.6*(255-pixel), 255)  for pixel in row] for row in imghsv[:,:,2]]
	

	imgBGR = cv.cvtColor(imghsv, cv.COLOR_HSV2BGR)
	picturename = os.path.basename(filename)
	print("picturename", picturename )
	cv.imwrite(outputdir+picturename, imgBGR, [cv.IMWRITE_JPEG_QUALITY, 100])
	print("Picture:", filename, "done")
	# cv.imshow('contrast', imgGRAY)
	# cv.waitKey(1000)
