import cv2 as cv
import numpy as np
import argparse
import glob
import os
import math
import time


def rotate_image(mat, angle):


	height, width, _ = mat.shape # image shape has 3 dimensions
	image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

	rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)

	# rotation calculates the cos and sin, taking absolutes of those.
	abs_cos = abs(rotation_mat[0,0]) 
	abs_sin = abs(rotation_mat[0,1])

	# find the new width and height bounds
	bound_w = int(height * abs_sin + width * abs_cos)
	bound_h = int(height * abs_cos + width * abs_sin)

	# subtract old image center (bringing image back to origo) and adding the new image center coordinates
	rotation_mat[0, 2] += bound_w/2 - image_center[0]
	rotation_mat[1, 2] += bound_h/2 - image_center[1]

	# rotate image with the new bounds and translated rotation matrix
	rotated_mat = cv.warpAffine(mat, rotation_mat, (bound_w, bound_h))
	return rotated_mat

# parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! ')
# parser.add_argument('--input', help='Path to input image.', default='./')
# parser.add_argument('--output', help='Path to input image.', default='./hole_filling/')
# parser.add_argument('--cuda', help='whether to use CUDA.', default=0)
# args = parser.parse_args()


# inputdir = args.input
# outputdir = args.output

# for filename in glob.glob(inputdir+"ortho_no_car.jpeg"):

# 	image = cv.imread(filename)
# 	i=0
# 	for y in range(0, image.shape[0], 10):
# 	    for x in range(0, image.shape[1], 10):
# 	    	if(image.shape[0]-666-y>=0 and image.shape[1]-666-x>=0):
# 		    	crop_image = image[y:666+y,x:666+x]
# 		    	picturename = os.path.basename(filename)
# 		    	# print("picturename", picturename )
# 		    	cv.imwrite(outputdir+str(i)+'_'+picturename, crop_image, [cv.IMWRITE_JPEG_QUALITY, 100])
# 		    	# print("Picture:", filename, "done")
# 		    	# cv.imshow('modified', crop_image)
# 		    	# cv.waitKey(1000)
# 		    	i += 1
text_file = open("tf.txt", "r")
lines = text_file.readlines()
total_image_number = len(lines)
ortho_no_car = cv.imread("./image/ortho_no_car.jpg")

for i in range(total_image_number):
	pic_name = lines[i].split()[0]
	tf_x = int(float(lines[i].split()[1]))
	tf_y = int(float(lines[i].split()[2]))
	tf_rotation = float(lines[i].split()[3])
	tf_scale = float(lines[i].split()[4])
	temp = cv.imread("./image/"+pic_name+".jpg")
	temp_r = rotate_image(temp, -tf_rotation)
	temp_width = int(temp_r.shape[1]*tf_scale)
	temp_hight = int(temp_r.shape[0]*tf_scale)
	temp_dim = (temp_width, temp_hight)
	temp_resize = cv.resize(temp_r, temp_dim)

	if(temp_resize.shape[0] % 2):
		image_gt = ortho_no_car[tf_y-int(temp_resize.shape[0]/2)-1:tf_y+int(temp_resize.shape[0]/2),\
							tf_x-int(temp_resize.shape[1]/2)-1:tf_x+int(temp_resize.shape[1]/2)]
	else:
		image_gt = ortho_no_car[tf_y-int(temp_resize.shape[0]/2):tf_y+int(temp_resize.shape[0]/2),\
							tf_x-int(temp_resize.shape[1]/2):tf_x+int(temp_resize.shape[1]/2)]
	print("tf y = ", tf_y)
	print("tf x = ", tf_x)
	print("temp size y = ", int(temp_resize.shape[0]/2))
	print("temp size x = ", int(temp_resize.shape[1]/2))
	print("image shape = ", temp_resize.shape)						
	print("image gt shape = ", image_gt.shape)
	cv.imshow("resized",temp_resize)
	cv.imshow("ortho",image_gt)
	cv.waitKey(5)
	cv.imwrite('./data_train/ground/'+pic_name+".jpg", temp_resize, [cv.IMWRITE_JPEG_QUALITY, 100])
	cv.imwrite('./data_train/aerial/'+pic_name+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])







