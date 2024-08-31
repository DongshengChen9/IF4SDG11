#%%
import os
from osgeo import gdal
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import alexnet, vgg16_bn
import time
import csv
import math
windowSize = 75
step = 10
#----------------------Gdal---------------------------------#
tif_path="./jm_buffer200.tif"
png_path=""
csv_path=""
tifList=os.listdir(tif_path)
pngList=os.listdir(png_path)
for file_index in range(len(tifList)):
	tif_file = tif_path + tifList[file_index]
	png_file = png_path + pngList[file_index]
	out_file = csv_path + tifList[file_index][0:-3] + "csv"
	dataset=gdal.Open(tif_file)
	adfGeoTransform = dataset.GetGeoTransform()
	im_width = dataset.RasterXSize
	im_height = dataset.RasterYSize
	im_bands = dataset.RasterCount
	print("rows:%d,cols:%d,bands:%d",im_height,im_width,im_bands)
	im_geotrans = dataset.GetGeoTransform()
	band=dataset.GetRasterBand(1)
	xMatrix = np.zeros((im_height,im_width))
	yMatrix = np.zeros((im_height,im_width))
	file= open(out_file,'w') 
	writer = csv.writer(file,lineterminator='\n')
	for i in range(0,im_height-windowSize+1,step):
		for j in range(0,im_width-windowSize+1,step):
			#print(str(i)+","+str(j))
			tmpTiff=band.ReadAsArray(j,i,windowSize,windowSize)
			#print(i,j)
			if -1 in tmpTiff:
				continue
			centerRow=i+math.floor(windowSize/2)
			centerCol=j+math.floor(windowSize/2)
			px = adfGeoTransform[0] + centerCol * adfGeoTransform[1] + centerRow * adfGeoTransform[2]
			py = adfGeoTransform[3] + centerCol * adfGeoTransform[4] + centerRow * adfGeoTransform[5]
			xMatrix[centerRow][centerCol]=round(px,6)
			yMatrix[centerRow][centerCol]=round(py,6)
	del dataset
	#file.close()
	
	# ------------------------- Input options ----------------------------- #
	# device = torch.device('cuda:0')  # for gpu
	device = torch.device('cuda:0')
	
	# ------------------------- Load data ----------------------------- #
	width = height = windowSize
	test_transform = transforms.Compose([
		transforms.Resize((width, height)),  # need PIL as input
		transforms.ToTensor(),  # normalize + transpose channel + convert type
	])
	Image.MAX_IMAGE_PIXELS = 1000000000
	img = Image.open(png_file)
	
	
	# ------------------------- Network ----------------------------- #
	# net = alexnet(pretrained=True)
	net = vgg16_bn(pretrained=True)
	net.to(device)
	print(net)
	net.eval()
	time_start = time.time()
	#--------------------------- Process ----------------------------------#
	index = 0
	for i in range(0,im_height-windowSize+1,step):
		for j in range(0,im_width-windowSize+1,step):
			centerRow=i+math.floor(windowSize/2)
			centerCol=j+math.floor(windowSize/2)
			x = xMatrix[centerRow][centerCol]
			y = yMatrix[centerRow][centerCol]
			if x == 0 or y == 0:
				if x != y:
					print("error!"+str(i)+","+str(j))
				continue
			tmpImg=img.crop((j,i,j+windowSize,i+windowSize))
			img_tsr = test_transform(tmpImg)
			#print(img_tsr.shape)
			index = index + 1
			if index%10000 == 0:
				print(index)
			inputs = img_tsr.unsqueeze(0)  
			inputs = inputs.to(device)
			feats = net.features(inputs)
			feats = F.adaptive_avg_pool2d(feats, output_size=(1, 1))
			feats = feats.squeeze()
			feats = feats.data.cpu().numpy()
			feats = feats.reshape(1,-1)
			temp = np.array([index,x,y,centerRow,centerCol])
			writer.writerow(np.append(temp,feats))
	
	file.close()
	img.close()
##############
time_end = time.time()
print(time_end - time_start)
