import cv2 as cv
import numpy as np
import argparse
import glob
import os
# import numba

parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! ')
parser.add_argument('--input', help='Path to input image.', default='../ground_pics/')
parser.add_argument('--output', help='Path to input image.', default='../ncc_output/')
args = parser.parse_args()

inputdir = args.input
outputdir = args.output



# Rotates an image (angle in degrees) and expands image to avoid cropping

def rotate_image(mat, angle):


	height, width = mat.shape # image shape has 3 dimensions
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


def NCC(img,temp,x,y):
	[H,W] = temp.shape
	
	sum_img = float(0)
	sum_temp = float(0)
	sum_2 = float(0)

	for i in range(W):
		for j in range(H):
			if(temp[j,i] != 255 & img[y+j, x+i] !=0):
				sum_img = sum_img + float(img[y+j,x+i])**2
				sum_temp = sum_temp + float(temp[j,i])**2
				sum_2 = sum_2 + (2-abs((W/2)-i)/(W/2)*abs((H/2)-j)/(H/2))*float(img[y+j,x+i])*float(temp[j,i])
			# sum_2 = sum_2 + float(img(y+j,x+i))*float(temp(j,i));

	val = sum_2/np.sqrt(float(sum_img)*float(sum_temp))
	return val


# start of the main loop

for filename in glob.glob(inputdir+"183_contrast.jpg"):


	# read souce image and convert it to gray
	img = cv.imread("../ortho_no_car.jpeg")
	img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	[img_height,img_width] = img_g.shape


	# read template imgage and convert it to gray, rotation and resize
	temp = cv.imread(filename)
	temp_g = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
	temp_r = rotate_image(temp_g, -93.5)
	# resize it
	scaling_factor = 3.54
	temp_width = int(temp_r.shape[1]*scaling_factor)
	temp_hight = int(temp_r.shape[0]*scaling_factor)
	temp_dim = (temp_width, temp_hight)
	temp_r = cv.resize(temp_r, temp_dim)

	# start of the ncc process
	regionXmin=13
	regionXmax=686
	regionYmin=13
	regionYmax=686

	maxrotation=1
	angle_resolution = 1
	val_max = -1
	xp = 0
	yp = 0
	number = 0
	totalcomputation = 0
	valmaxlist=np.zeros([1,maxrotation], dtype=float)
	coordinatelists=np.zeros([2,maxrotation], dtype=float)

	for rotationtest in range(maxrotation):
		temp_r1 = rotate_image(temp_r, -(rotationtest)*angle_resolution)
		[temp_H,temp_W] = temp_r1.shape
		totalcomputation = totalcomputation+(regionXmax-regionXmin-temp_W+1)*(regionYmax-regionYmin-temp_H+1)

	print("the total computation times", totalcomputation)


	for rotation in  range(maxrotation):


		temp_g = rotate_image(temp_r, -(rotation)*angle_resolution)
		
		
		[temp_H,temp_W] = temp_g.shape
		
		dis = np.ones([regionYmax-temp_H-regionYmin+1,regionXmax-temp_W-regionXmin+1], dtype=float)
		
		for y in range(regionYmin-1, regionYmax-temp_H):
			for x in range(regionXmin-1, regionXmax-temp_W):
				val = NCC(img_g,temp_g,x,y)
				number = number + 1
				progess = 100*number/totalcomputation
				dis[y-regionYmin+1,x-regionXmin+1]=val
				print("Progress=", progess)
				if val > val_max:
					val_max = val
					xp = x
					yp = y
					angle=rotation
		valmaxlist[0,rotation] = val_max
		print("xp= ", xp)
		print("yp= ", yp)
		coordinatelists[0,rotation]=xp
		coordinatelists[1,rotation]=yp

	print("coordinate_list ", coordinatelists)
	print("distance", dis)



	



	picturename = os.path.basename(filename)
	print("picturename", picturename )
	cv.imwrite(outputdir+picturename, image, [cv.IMWRITE_JPEG_QUALITY, 100])
	print("Picture:", filename, "done")
	cv.imshow('modified', image)
	cv.waitKey(10000)
