# import cv2
# path = './data/512/train/images/1_1.png'
# img = cv2.imread(path)
# img =img*255
# cv2.imshow('1',img)
# cv2.waitKey(0)

import os
from PIL import Image

img_path = './dataset/512/val/labels/'
img_path_new = './dataset/512/val/labels255/'
# if not os.path.isdir(img_path_new):
#     # 建立一个新的文件夹
#     os.mkdir(img_path_new)

# 给文件家里面的文件排序
img_list = sorted(os.listdir(img_path))
print(img_list)

x = 0
y = 0
# set() 函数创建一个无序不重复元素集
pix_set = set()

'''
for image in img_list:
	img_name = str(image)
	img = Image.open(os.path.join(img_path,image))
	img_weight = img.size[0]
	img_hight = img.size[1]
	w_range = range(img_weight)
	h_range = range(img_hight)
	for x in w_range:
		for y in h_range:
			pix = img.getpixel((x,y))
			pix_set.add(pix)
print("The pixel values in images include:")
for i in pix_set:
  print(i)
'''

for image in img_list:
	img_name = str(image)
	img = Image.open(os.path.join(img_path,image))
	img_weight = img.size[0]
	img_hight = img.size[1]
	w_range = range(img_weight)
	h_range = range(img_hight)
	for x in w_range:
		for y in h_range:
            # 得到像素值
			pix = img.getpixel((x,y))
			pix_set.add(pix)
			if pix == 0: # background
				pix = img.putpixel((x,y),0)
			elif pix > 0 :
                                pix = img.putpixel((x,y),255)

#	(temp_name, temp_extention) = os.path.splitext(image)
#	img_name_new = temp_name + '_new' + temp_extention
	img_new = os.path.join(img_path_new, img_name)
	img.save(img_new)
	print (image + " pixel values changed and saved to " + img_name)


