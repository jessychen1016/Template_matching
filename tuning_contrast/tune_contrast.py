from __future__ import print_function
from builtins import input
import cv2 as cv
import numpy as np
import argparse
# Read image given by user
parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! tutorial.')
parser.add_argument('--input', help='Path to input image.', default='./195.jpg')
args = parser.parse_args()
image = cv.imread(cv.samples.findFile(args.input))
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
cv.imshow('Original Image', image)
cv.waitKey(1000)
imghsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)


imghsv[:,:,2] = [[max(0.5*pixel, 0) if pixel < 60 else min(pixel+0.8*(255-pixel), 255) for pixel in row] for row in imghsv[:,:,2]]
cv.imshow('contrast', cv.cvtColor(imghsv, cv.COLOR_HSV2BGR))
cv.waitKey(20000)
