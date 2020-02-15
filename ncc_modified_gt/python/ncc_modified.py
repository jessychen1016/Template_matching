import cv2 as cv
import numpy as np
import argparse
import glob
import os
import utility_ncc
import math
from numba import cuda
import time

parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! ')
parser.add_argument('--input', help='Path to input image.', default='../ground_pics/')
parser.add_argument('--output', help='Path to input image.', default='../ncc_output/')
parser.add_argument('--cuda', help='whether to use CUDA.', default=0)
args = parser.parse_args()

inputdir = args.input
outputdir = args.output
use_cuda = bool(args.cuda)
print("Do We Use CUDA?", use_cuda)

# start of the main loop

for filename in glob.glob(inputdir+"183_contrast.jpg"):


	# read souce image and convert it to gray
	img = cv.imread("../ortho_no_car.jpeg")
	img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	[img_height,img_width] = img_g.shape


	# read template imgage and convert it to gray, rotation and resize
	temp = cv.imread(filename)
	temp_g = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
	temp_r = utility_ncc.rotate_image(temp_g, -93.5)
	# resize it
	scaling_factor = 3.54
	temp_width = int(temp_r.shape[1]*scaling_factor)
	temp_hight = int(temp_r.shape[0]*scaling_factor)
	temp_dim = (temp_width, temp_hight)
	temp_r = cv.resize(temp_r, temp_dim)

	# start of the ncc process by initializing some of the veriables

	regionXmin=13
	regionXmax=700
	regionYmin=13
	regionYmax=700

	maxrotation=1
	angle_resolution = 1
	val_max = -1
	xp = 0
	yp = 0
	number = 0
	totalcomputation = 0
	valmaxlist=np.zeros([1,maxrotation], dtype=float)
	coordinatelists=np.zeros([2,maxrotation], dtype=int)

	# params for CUDA
	val_cuda=float(0)
	sum_img = float(0)
	sum_temp = float(0)
	sum_2 = float(0)
	x_cuda = int(0)
	y_cuda = int(0)

	# calculate the number of total pixels
	for rotationtest in range(maxrotation):
		temp_r1 = utility_ncc.rotate_image(temp_r, -(rotationtest)*angle_resolution)
		[temp_H,temp_W] = temp_r1.shape
		totalcomputation = totalcomputation+(regionXmax-regionXmin-temp_W+1)*(regionYmax-regionYmin-temp_H+1)

	print("the total computation times", totalcomputation)

	# start calculating
	for rotation in  range(maxrotation):


		temp_g = utility_ncc.rotate_image(temp_r, -(rotation)*angle_resolution)
		
		
		[temp_H,temp_W] = temp_g.shape
		
		dis = np.ones([regionYmax-temp_H-regionYmin+1,regionXmax-temp_W-regionXmin+1], dtype=float)
		
		# set initial device parameters for CUDA
		if(use_cuda):
			threadsperblock = (32, 32)
			blockspergrid_x = int(math.ceil(temp_g.shape[0] / threadsperblock[0]))
			blockspergrid_y = int(math.ceil(temp_g.shape[1] / threadsperblock[1]))
			blockspergrid = (blockspergrid_x, blockspergrid_y)
			# pass some scalar values to the kernal
			val_cuda_global_mem = cuda.to_device(val_cuda)
			sum_img_global_mem = cuda.to_device(sum_img)
			sum_temp_global_mem = cuda.to_device(sum_temp)
			sum_2_global_mem = cuda.to_device(sum_2)
			#pass the temp array to cuda memory
			temp_cuda_global_mem = cuda.to_device(temp_g)
			img_cuda_global_mem = cuda.to_device(img_g)
			x_cuda_global_mem = cuda.to_device(x_cuda)
			y_cuda_global_mem = cuda.to_device(y_cuda)

		loop_start_time = time.time()
		for y in range(regionYmin-1, regionYmax-temp_H):
			for x in range(regionXmin-1, regionXmax-temp_W):
				if(use_cuda):
					# start_time=time.time()
					
					sum_img_global_mem[0], sum_temp_global_mem[0], sum_2_global_mem[0] = 0,0,0 
					x_cuda_global_mem[0], y_cuda_global_mem[0] = x,y
					# to call the kernal
					middle_time=time.time()
					utility_ncc.cudaNCC[blockspergrid,threadsperblock](img_cuda_global_mem,temp_cuda_global_mem,\
																		x_cuda_global_mem ,y_cuda_global_mem, sum_img_global_mem,\
																		sum_temp_global_mem, sum_2_global_mem)
					val = sum_2_global_mem[0]/np.sqrt(float(sum_img_global_mem[0])*float(sum_temp_global_mem[0]))
					# print("cuda_val", val)
					# val = val_cuda
					# end_time=time.time()

					# print("passing time= ", middle_time- start_time)
					# print("progressing time= ", end_time- middle_time)
					# print("total time= ", end_time- start_time)
				else:
					val = utility_ncc.NCC(img_g,temp_g,x,y)
				number = number + 1
				progess = 100*number/totalcomputation
				# dis[y-regionYmin+1,x-regionXmin+1]=val
				print("Progress=", progess)
				if val > val_max:
					val_max = val
					xp = x
					yp = y
					angle=rotation
		

		loop_finish_time = time.time()
		valmaxlist[0,rotation] = val_max
		print("xp= ", xp)
		print("yp= ", yp)
		coordinatelists[0,rotation]=xp
		coordinatelists[1,rotation]=yp
		print("The total Time Consumption is ", loop_finish_time-loop_start_time)

	print("coordinate_list ", coordinatelists)
	# print("distance", dis)

	picturename = os.path.basename(filename)
	# print("picturename", picturename )
	# cv.imwrite(outputdir+picturename, image, [cv.IMWRITE_JPEG_QUALITY, 100])
	# print("Picture:", filename, "done")
	# cv.imshow('modified', image)
	# cv.waitKey(10000)
