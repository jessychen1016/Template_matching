import numpy as np
import cv2 as cv
from numba import cuda



# use cuda to calculate the ncc value
@cuda.jit
def cudaNCC(img_original,temp,x,y,sum_img,sum_temp,sum_2):
	[H,W] = temp.shape
	# print("sum2= ",sum_2[0])
	i,j = cuda.grid(2)
	# print("img[",i,",",j,"]=",x[0], y[0])

	if (j < temp.shape[0] and i < temp.shape[1]):
		if(temp[j,i] != 255 & img_original[j+y[0],i+x[0]] !=0):
				

				## use cuda.atomic computation to avoid conflict between threads
				cuda.atomic.add(sum_img,0,float(img_original[j+y[0],i+x[0]])**2)
				cuda.atomic.add(sum_temp,0,float(temp[j,i])**2)
				cuda.atomic.add(sum_2,0,float(2-abs((W/2)-i)/(W/2)*abs((H/2)-j)/(H/2))*float(img_original[j+y[0],i+x[0]])*float(temp[j,i]))
				
				## add without sync
				# sum_img[0] = sum_img[0] + float(img_original[j+y[0],i+x[0]])**2
				# sum_temp[0] = sum_temp[0] + float(temp[j,i])**2
				# sum_2[0] = sum_2[0] + float(2-abs((W/2)-i)/(W/2)*abs((H/2)-j)/(H/2))*float(img_original[j+y[0],i+x[0]])*float(temp[j,i])

				# sum_2[0] = sum_2[0] + float(img_original[j+y[0],i+x[0]])*float(temp[j,i])
				# print("the 556,443 f img= ", img[556,443])
				# print("sum2= ",sum_2[0])


# calculate the ncc value without cuda
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
				# sum_2 = sum_2 + float(img(y+j,x+i))*float(temp(j,i))
	val = sum_2/np.sqrt(float(sum_img)*float(sum_temp))
	return val


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