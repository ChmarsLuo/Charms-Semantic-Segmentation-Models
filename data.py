from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
import glob
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

############################################
#             数据处理                      #
############################################
def adjustData(img, mask, flag_multi_class, num_class):
    if(flag_multi_class):
        #img = img /255.
        mask = mask[:, :, :, 0] if (len(mask.shape)==4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,)) # (512, 512, 3, 2)
        for i in range(num_class):
            # index = np.where(mask == i) # 元组
            # index_mask = (index[0], index[1], index[2], np.zeros(len(index[0]), dtype=np.int64) + i) if (
            #             len(mask.shape) == 4) else (index[0], index[1], np.zeros(len(index[0]), dtype=np.int64) + i)
            # new_mask[index_mask] = 1
            new_mask[mask == i, i] = 1 # 将平面的mask的每类，都单独变成一层
            new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2],
                                             new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (
            new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
            mask = new_mask


    elif(np.max(img) > 1):
        img = img / 255.
        mask = mask / 255.
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] =1

    return (img,mask)
# adjustData(img, img, 1, 2 )

def trainGenerator(batch_size, train_path, image_folder, mask_folder,aug_dict,
                   image_save_data_dir, mask_save_data_dir, image_color_mode = "rgb",
                   mask_color_mode = "grayscale",image_save_prefix  = "image_",mask_save_prefix  = "mask_",
                   flag_multi_class = False,num_class = 2,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    mask_color_mode = "grayscale"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        # save_to_dir = None,
        save_to_dir = image_save_data_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        # save_to_dir=None,
        save_to_dir = mask_save_data_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator) # 变成一对一对的啦
    for (img,mask) in train_generator:

        img,mask = adjustData(img,mask, flag_multi_class, num_class)
        # print(img.shape) # (4, 416, 416, 3)
        # print(mask.shape) # 输出的是这个样子呀(4, 416, 416, 1)

        yield (img,mask)


def testGenerator(test_path,num_image = 110,target_size = (256,256,3),flag_multi_class = False,as_gray = True):
    imgs = os.listdir(test_path)
    for im in imgs:
        # img = io.imread(os.path.join(test_path,im),as_gray = as_gray)
        # img = io.imread(os.path.join(test_path,im))
        img = cv2.imread(os.path.join(test_path, im))
        #print(img)
        img = img / 255.0
        img = trans.resize(img,target_size)
        # img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        #img = np.expand_dims(img,3)
        yield img
    # for i in range(num_image):
    #     img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
    #     img = img / 255
    #     img = trans.resize(img,target_size)
    #     img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
    #     img = np.reshape(img,(1,)+img.shape)
    #     yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,)) ##变成RGB空间，因为其他颜色只能再RGB空间才会显示
    for i in range(num_class): #为不同类别涂上不同的颜色，color_dict[i]是与类别数有关的颜色，img_out[img == i,:]是img_out在img中等于i类的位置上的点
        img_out[img == i,:] = color_dict[i]
    return img_out / 255.0
#####上面函数是给出测试后的输出之后，为输出涂上不同的颜色，多类情况下才起作用，两类的话无用


def saveResult(save_path,npyfile,names,flag_multi_class = False,num_class = 2):
    # for i,item in enumerate(npyfile):
    #     img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
    #     print(img)
    #     io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)

    for i,item in enumerate(npyfile):

        if flag_multi_class:

            img = labelVisualize(num_class,COLOR_DICT,item)
        else:
            img = item[:,:,0]
            # img = img.astype(np.uint8)
            print(img)
            img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_NEAREST)
            print(np.max(img), np.min(img))
            img[img>0.5] = 1
            img[img<=0.5] = 0

            print(np.max(img), np.min(img))
        io.imsave(os.path.join(save_path,"%s"%names[i]),img)
        # name = names[i].split('.')[0]#多类的话就图成彩色，非多类（两类）的话就是黑白色
        # io.imsave(os.path.join(save_path,"%s.tif"%name),img_as_ubyte(img))

# def saveResult(save_path,npyfile,names,flag_multi_class=False):
#     for i,item in enumerate(npyfile):
#         if flag_multi_class:
#             img = item
#             img_out = np.zeros(img[:, :, 0].shape + (3,))
#             for row in range(img.shape[0]):
#                 for col in range(img.shape[1]):
#                     index_of_class = np.argmax(img[row, col])
#                     # img_out[row, col] = COLOR_DICT[index_of_class]
#                     img_out[row, col] = index_of_class #预测的时候。评估数据不能带有颜色的
#             img = img_out.astype(np.uint8) #很重要的一步
#             # io.imsave(os.path.join(save_path, '%s_predict.png' % i), img)
#             io.imsave(os.path.join(save_path, "%s"%names[i]), img)
#         else:
#             img = item[:, :, 0]
#             img[img > 0.5] = 1
#             img[img <= 0.5] = 0
#             img = img * 255.
#             io.imsave(os.path.join(save_path, "%s" % names[i]), img)
#             # io.imsave(os.path.join(save_path, '%s_predict.png' % i), img)
