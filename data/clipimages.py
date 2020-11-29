# -*- coding: utf-8 -*-
import os
import cv2
from osgeo import gdal
import numpy as np

def read_img(filename):
    dataset=gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)

    del dataset 
    return im_proj,im_geotrans,im_width, im_height,im_data

def write_img(filename,im_proj,im_geotrans,im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape 

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])


def gdal_image_clip(inpath, outpath, new_width=500, stride=200):
    test_im_dir = os.listdir(inpath)
    for name in test_im_dir:
        if name[-4:] == '.png':
            print("dealing the ",name," ...")
            img = os.path.join(inpath, name)           
            im_proj,im_geotrans,im_width, im_height,im_data = read_img(img)
            new_w = im_width
            new_h = im_height
            extent_data = im_data
            # print(extent_data.shape)

            count = 0
            i = 0
            num_ = 0

            filename = name[:-4]
            while i in range(new_h):
                j=0
                if (new_h-i) >=new_width:
                    while j in range(new_w):          
                        if (new_w-j) >=new_width:
                            num_=num_+1
                            im_data_m=extent_data[:,i:i+new_width,j:j+new_width]
                            patch_path = os.path.join(outpath, filename + '_' + str(num_) + '.png')
                            im_data_m = im_data_m.transpose(1,2,0)
                            cv2.imwrite(patch_path, im_data_m, [int(cv2.cv2.IMWRITE_PNG_COMPRESSION),0])
                            # write_img(os.path.join(outpath, filename + '_' + str(num_) + '.tif'), im_proj, im_geotrans, im_data_m)
                            j=j+stride
                            
                        if (new_w-j) <new_width:
                            num_=num_+1
                            im_data_m=extent_data[:,i:i+new_width,new_w-new_width:new_w]
                            patch_path = os.path.join(outpath, filename + '_' + str(num_) + '.png')
                            im_data_m = im_data_m.transpose(1,2,0)
                            cv2.imwrite(patch_path, im_data_m, [int(cv2.cv2.IMWRITE_PNG_COMPRESSION),0])
                            # write_img(os.path.join(outpath, filename + '_' + str(num_) + '.tif'), im_proj, im_geotrans, im_data_m)
                            j=new_w+1
                            
                    i=i+stride

                else :
                    while j in range(new_w):
                        if  (new_w-j) >=new_width:
                            num_=num_+1
                            im_data_m=extent_data[:,new_h-new_width:new_h,j:j+new_width]
                            patch_path = os.path.join(outpath, filename + '_' + str(num_) + '.png')
                            im_data_m = im_data_m.transpose(1,2,0)
                            cv2.imwrite(patch_path, im_data_m, [int(cv2.cv2.IMWRITE_PNG_COMPRESSION),0])
                            # write_img(os.path.join(outpath, filename + '_' + str(num_) + '.tif'), im_proj, im_geotrans, im_data_m)
                            j=j+stride
                            
                        if (new_w-j) <new_width:
                            num_=num_+1
                            im_data_m=extent_data[:,new_h-new_width:new_h,new_w-new_width:new_w]
                            patch_path = os.path.join(outpath, filename + '_' + str(num_) + '.png')
                            im_data_m = im_data_m.transpose(1,2,0)
                            cv2.imwrite(patch_path, im_data_m, [int(cv2.cv2.IMWRITE_PNG_COMPRESSION),0])
                            # write_img(os.path.join(outpath, filename + '_' + str(num_) + '.tif'), im_proj, im_geotrans, im_data_m)
                            j=new_w+1
                            
                    i=new_h+1


def gdal_label_clip(inpath, outpath, new_width=500, stride=200):
    test_im_dir = os.listdir(inpath)
    for name in test_im_dir:
        if name[-4:] == '.png':
            print("dealing the ",name," ...")
            img = os.path.join(inpath, name)           
            im_proj,im_geotrans,im_width, im_height,im_data = read_img(img)
            new_w = im_width
            new_h = im_height
            extent_data = im_data
            # print(extent_data.shape)

            count = 0
            i = 0
            num_ = 0

            filename = name[:-4]
            while i in range(new_h):
                j=0
                if (new_h-i) >=new_width:
                    while j in range(new_w):          
                        if (new_w-j) >=new_width:
                            num_=num_+1
                            # im_data_m=extent_data[:,i:i+new_width,j:j+new_width]
                            im_data_m = extent_data[ i:i + new_width, j:j + new_width]
                            # print(im_data_m)
                            patch_path = os.path.join(outpath, filename + '_' + str(num_) + '.png')
                            print(patch_path)
                            # im_data_m = im_data_m.transpose(1,2,0)
                            cv2.imwrite(patch_path, im_data_m, [int(cv2.cv2.IMWRITE_PNG_COMPRESSION),0])
                            # write_img(os.path.join(outpath, filename + '_' + str(num_) + '.tif'), im_proj, im_geotrans, im_data_m)
                            j=j+stride
                            
                        if (new_w-j) <new_width:
                            num_=num_+1
                            # im_data_m=extent_data[0,i:i+new_width,new_w-new_width:new_w]
                            im_data_m = extent_data[ i:i + new_width, new_w - new_width:new_w]
                            patch_path = os.path.join(outpath, filename + '_' + str(num_) + '.png')
                            # im_data_m = im_data_m.transpose(1,2,0)
                            cv2.imwrite(patch_path, im_data_m, [int(cv2.cv2.IMWRITE_PNG_COMPRESSION),0])
                            # write_img(os.path.join(outpath, filename + '_' + str(num_) + '.tif'), im_proj, im_geotrans, im_data_m)
                            j=new_w+1
                            
                    i=i+stride

                else :
                    while j in range(new_w):
                        if  (new_w-j) >=new_width:
                            num_=num_+1
                            # im_data_m=extent_data[0,new_h-new_width:new_h,j:j+new_width]
                            im_data_m = extent_data[ new_h - new_width:new_h, j:j + new_width]
                            patch_path = os.path.join(outpath, filename + '_' + str(num_) + '.png')
                            # im_data_m = im_data_m.transpose(1,2,0)
                            cv2.imwrite(patch_path, im_data_m, [int(cv2.cv2.IMWRITE_PNG_COMPRESSION),0])
                            # write_img(os.path.join(outpath, filename + '_' + str(num_) + '.tif'), im_proj, im_geotrans, im_data_m)
                            j=j+stride
                            
                        if (new_w-j) <new_width:
                            num_=num_+1
                            # im_data_m=extent_data[0,new_h-new_width:new_h,new_w-new_width:new_w]
                            im_data_m = extent_data[ new_h - new_width:new_h, new_w - new_width:new_w]
                            patch_path = os.path.join(outpath, filename + '_' + str(num_) + '.png')
                            # im_data_m = im_data_m.transpose(1,2,0)
                            cv2.imwrite(patch_path, im_data_m, [int(cv2.cv2.IMWRITE_PNG_COMPRESSION),0])
                            # write_img(os.path.join(outpath, filename + '_' + str(num_) + '.tif'), im_proj, im_geotrans, im_data_m)
                            j=new_w+1
                            
                    i=new_h+1


if __name__ == '__main__':
    print('Clip image...')
    in_img_path = 'C:\\Users\\Charm Luo\\Desktop\\my-data\\duofenlei\\DEN-SENET\\camvid\\valid_images\\'
    out_img_path = 'C:\\Users\\Charm Luo\\Desktop\\my-data\\duofenlei\\DEN-SENET\\camvid\\valid_images_cut\\'
    gdal_image_clip(in_img_path, out_img_path, new_width=256, stride=256)

    print('Clip label...')
    in_label_path = 'C:\\Users\\Charm Luo\\Desktop\\my-data\\duofenlei\\DEN-SENET\\camvid\\valid_labels\\'
    out_label_path = 'C:\\Users\\Charm Luo\\Desktop\\my-data\\duofenlei\\DEN-SENET\\camvid\\valid_labels_cut\\'
    gdal_label_clip(in_label_path, out_label_path, new_width=256, stride=256)