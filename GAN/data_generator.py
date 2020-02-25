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


def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
def salt_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
def pepper_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            else:
                output[i][j] = image[i][j]
    return output

# parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! ')
# parser.add_argument('--input', help='Path to input image.', default='./')
# parser.add_argument('--output', help='Path to input image.', default='./hole_filling/')
# parser.add_argument('--cuda', help='whether to use CUDA.', default=0)
# args = parser.parse_args()
# inputdir = args.input
# outputdir = args.output



multi_image = True
add_mask = True
change_contrast = True
mask_type = "Gaussion"

if(multi_image):


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
		print("image number= ", pic_name)
		print("tf y = ", tf_y)
		print("tf x = ", tf_x)
		print("temp size y = ", int(temp_resize.shape[0]/2))
		print("temp size x = ", int(temp_resize.shape[1]/2))
		print("image shape = ", temp_resize.shape)						
		print("image gt shape = ", image_gt.shape)

		if(add_mask):
			if(mask_type == "Gaussion"):
				image_masked = cv.GaussianBlur(temp_resize,(5,5),0,0)
			if(mask_type == "SaltPepper"):
				image_masked = sp_noise(temp_resize, 0.01)
			if(mask_type == "Salt"):
				image_masked = salt_noise(temp_resize, 0.01)
			if(mask_type == "Pepper"):
				image_masked = pepper_noise(temp_resize, 0.01)

			if(change_contrast):
				image_c1 = cv.convertScaleAbs(image_masked, alpha=0.25, beta=0)
				image_c2 = cv.convertScaleAbs(image_masked, alpha=0.5, beta=0)
				image_c3 = cv.convertScaleAbs(image_masked, alpha=1.25, beta=0)
				image_c4 = cv.convertScaleAbs(image_masked, alpha=1.5, beta=0)
				# cv.imshow("image_c1",image_c1)
				# cv.imshow("image_c2",image_c2)
				# cv.imshow("image_c3",image_c3)
				# cv.imshow("image_c4",image_c4)
				# cv.imshow(mask_type,image_masked)
				# cv.waitKey(5000)

			# cv.imshow("ortho_"+pic_name, image_gt)
			# cv.waitKey(5)
			cv.imwrite('./data_train/ground/'+pic_name+'_'+mask_type+".jpg", image_masked, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/ground/'+pic_name+'_'+mask_type+"_c1"+".jpg", image_c1, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/ground/'+pic_name+'_'+mask_type+"_c2"+".jpg", image_c2, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/ground/'+pic_name+'_'+mask_type+"_c3"+".jpg", image_c3, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/ground/'+pic_name+'_'+mask_type+"_c4"+".jpg", image_c4, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/aerial/'+pic_name+'_'+mask_type+"_c1"+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/aerial/'+pic_name+'_'+mask_type+"_c2"+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/aerial/'+pic_name+'_'+mask_type+"_c3"+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/aerial/'+pic_name+'_'+mask_type+"_c4"+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/aerial/'+pic_name+'_'+mask_type+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])
		else:
			if(change_contrast):
				image_c1 = cv.convertScaleAbs(temp_resize, alpha=0.25, beta=0)
				image_c2 = cv.convertScaleAbs(temp_resize, alpha=0.5, beta=0)
				image_c3 = cv.convertScaleAbs(temp_resize, alpha=1.25, beta=0)
				image_c4 = cv.convertScaleAbs(temp_resize, alpha=1.5, beta=0)
				# cv.imshow("image_c1",image_c1)
				# cv.imshow("image_c2",image_c2)
				# cv.imshow("image_c3",image_c3)
				# cv.imshow("image_c4",image_c4)
				# cv.imshow(mask_type,image_masked)
				# cv.waitKey(5000)
			cv.imwrite('./data_train/ground/'+pic_name+"_c1"+".jpg", image_c1, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/ground/'+pic_name+"_c2"+".jpg", image_c2, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/ground/'+pic_name+"_c3"+".jpg", image_c3, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/ground/'+pic_name+"_c4"+".jpg", image_c4, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/ground/'+pic_name+".jpg", temp_resize, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/aerial/'+pic_name+"_c1"+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/aerial/'+pic_name+"_c2"+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/aerial/'+pic_name+"_c3"+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/aerial/'+pic_name+"_c4"+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/aerial/'+pic_name+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])


else:
	text_file = open("tf.txt", "r")
	lines = text_file.readlines()
	total_image_number = len(lines)
	ortho_no_car = cv.imread("./image/ortho_no_car.jpg")

	for i in range(1):
		pic_name = lines[i].split()[0]
		tf_x = int(float(lines[i].split()[1]))
		tf_y = int(float(lines[i].split()[2]))
		tf_rotation = float(lines[i].split()[3])
		tf_scale = float(lines[i].split()[4])
		temp = cv.imread("./image/"+"171"+".jpg")
		temp_r = rotate_image(temp, -tf_rotation)
		temp_width = int(temp_r.shape[1]*tf_scale)
		temp_hight = int(temp_r.shape[0]*tf_scale)
		temp_dim = (temp_width, temp_hight)
		temp_resize = cv.resize(temp_r, temp_dim)

		cv.imwrite('./'+pic_name+".jpg", temp_resize, [cv.IMWRITE_JPEG_QUALITY, 100])
		# cv.imwrite('./data_train/aerial/'+pic_name+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])






