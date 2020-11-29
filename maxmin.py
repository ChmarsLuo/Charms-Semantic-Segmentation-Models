import numpy as np
import cv2
path = './output/build-1-result/22828930_15_1.png'
img = cv2.imread(path, 0)
max = np.max(img)
min = np.min(img)
print('===========',max)
print('+++++++++++',min)
print('___________',img)