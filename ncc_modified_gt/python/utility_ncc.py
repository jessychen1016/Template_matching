import numpy as np
import cv2 as cv
from numba import cuda



# use cuda to calculate the ncc value
@cuda.jit
def cudaNCC(img_original,temp,y,x,sum_img,sum_temp,sum_2):
	[H,W] = temp.shape
	# print("sum2= ",sum_2[0])
	i,j = cuda.grid(2)
	# print("img[",i,",",j,"]=",x[0], y[0])

	if (j < temp.shape[0] and i < temp.shape[1]):
		if(temp[j,i] != 255 & img_original[j+y[0],i+x[0]] !=0):
				

				## use cuda.atomic computation to avoid conflict between threads
				cuda.atomic.add(sum_img, (y-searching_boundary[0],x-searching_boundary[2]),float(img_original[j+y[0],i+x[0]])**2)
				cuda.atomic.add(sum_temp, (y-searching_boundary[0],x-searching_boundary[2]),float(temp[j,i])**2)
				cuda.atomic.add(sum_2,(y-searching_boundary[0],x-searching_boundary[2]),float(2-abs((W/2)-i)/(W/2)*abs((H/2)-j)/(H/2))*float(img_original[j+y[0],i+x[0]])*float(temp[j,i]))
				
				## add without sync
				# sum_img[0] = sum_img[0] + float(img_original[j+y[0],i+x[0]])**2
				# sum_temp[0] = sum_temp[0] + float(temp[j,i])**2
				# sum_2[0] = sum_2[0] + float(2-abs((W/2)-i)/(W/2)*abs((H/2)-j)/(H/2))*float(img_original[j+y[0],i+x[0]])*float(temp[j,i])

				# sum_2[0] = sum_2[0] + float(img_original[j+y[0],i+x[0]])*float(temp[j,i])
				# print("the 556,443 f img= ", img[556,443])
				# print("sum2= ",sum_2[0])



@cuda.jit
def cuda_searchingNCC(img_cuda_global_mem, temp_cuda_global_mem, \
						sum_img_global_mem, sum_temp_global_mem, sum_2_global_mem,\
						dis, searching_boundary,\
						blockspergrid_ncc,threadsperblock_ncc):

	y,x = cuda.grid(2)
	# start_time=time.time()
	
	val=-1

	if (y>=searching_boundary[0] and y< searching_boundary[1] and x>= searching_boundary[2] and x<searching_boundary[3]):

		sum_img_global_mem[y-searching_boundary[0],x-searching_boundary[2]]=0
		sum_temp_global_mem[y-searching_boundary[0],x-searching_boundary[2]]=0
		sum_2_global_mem[y-searching_boundary[0],x-searching_boundary[2]]=0
		# to call the kernal
		print("haaaaaaaaaaaaaaaaaaaaaa")
		cudaNCC[blockspergrid_ncc[0],threadsperblock_ncc[0]](img_cuda_global_mem,temp_cuda_global_mem,\
															y ,x, sum_img_global_mem,\
															sum_temp_global_mem, sum_2_global_mem)
		val = sum_2_global_mem[y-searching_boundary[0],x-searching_boundary[2]]\
					/np.sqrt(float(sum_img_global_mem[y-searching_boundary[0],x-searching_boundary[2]])\
					*float(sum_temp_global_mem[y-searching_boundary[0],x-searching_boundary[2]]))
		# print("cuda_val", val)
		# val = val_cuda
		# end_time=time.time()

		# print("passing time= ", middle_time- start_time)
		# print("progressing time= ", end_time- middle_time)
		# print("total time= ", end_time- start_time)

		# number = number + 1
		# progess = 100*number/totalcomputation
		dis[y-searching_boundary[0],x-searching_boundary[2]]=val
		# print("Progress=", progess)
		cuda.syncthreads()




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