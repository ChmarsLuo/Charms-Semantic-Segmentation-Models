from keras.models import *
from keras.layers import *

import keras.backend as K
import keras
IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1

def relu6(x):
	return K.relu(x, max_value=6)




def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):

	channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
	filters = int(filters * alpha)
	x = ZeroPadding2D(padding=(1, 1), name='conv1_pad', data_format=IMAGE_ORDERING  )(inputs)
	x = Conv2D(filters, kernel , data_format=IMAGE_ORDERING  ,
										padding='valid',
										use_bias=False,
										strides=strides,
										name='conv1')(x)
	x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
	return Activation(relu6, name='conv1_relu')(x)




def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
													depth_multiplier=1, strides=(1, 1), block_id=1):

	channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
	pointwise_conv_filters = int(pointwise_conv_filters * alpha)

	x = ZeroPadding2D((1, 1) , data_format=IMAGE_ORDERING , name='conv_pad_%d' % block_id)(inputs)
	x = DepthwiseConv2D((3, 3) , data_format=IMAGE_ORDERING ,
														 padding='valid',
														 depth_multiplier=depth_multiplier,
														 strides=strides,
														 use_bias=False,
														 name='conv_dw_%d' % block_id)(x)
	x = BatchNormalization(
			axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
	x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

	x = Conv2D(pointwise_conv_filters, (1, 1), data_format=IMAGE_ORDERING ,
										padding='same',
										use_bias=False,
										strides=(1, 1),
										name='conv_pw_%d' % block_id)(x)
	x = BatchNormalization(axis=channel_axis,
																name='conv_pw_%d_bn' % block_id)(x)
	return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)





def get_mobilenet_encoder( input_height=224 ,  input_width=224 , pretrained='imagenet' ):

	alpha=1.0
	depth_multiplier=1
	dropout=1e-3


	img_input = Input(shape=(input_height,input_width , 3 ))


	x = _conv_block(img_input, 32, alpha, strides=(2, 2))
	x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
	f1 = x

	x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
														strides=(2, 2), block_id=2)
	x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
	f2 = x

	x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
														strides=(2, 2), block_id=4)
	x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
	f3 = x

	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
														strides=(2, 2), block_id=6)
	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
	f4 = x

	x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
														strides=(2, 2), block_id=12)
	x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
	f5 = x

	return img_input , [f1 , f2 , f3 , f4 , f5 ]


import tensorflow as tf
################################################
#                  psp_mobilenet               #
################################################
def resize_image(inp, s, data_format):


    return Lambda(
        lambda x: tf.image.resize_images(
            x, (K.int_shape(x)[1] * s[0], K.int_shape(x)[2] * s[1]))
    )(inp)


def pool_block(feats, pool_factor):
    if IMAGE_ORDERING == 'channels_first':
        h = K.int_shape(feats)[2]
        w = K.int_shape(feats)[3]
    elif IMAGE_ORDERING == 'channels_last':
        h = K.int_shape(feats)[1]
        w = K.int_shape(feats)[2]

    # strides = [18,18],[9,9],[6,6],[3,3]
    pool_size = strides = [int(np.round(float(h) / pool_factor)), int(np.round(float(w) / pool_factor))]

    # 进行不同程度的平均
    x = AveragePooling2D(pool_size, data_format=IMAGE_ORDERING, strides=strides, padding='same')(feats)

    # 进行卷积
    x = Conv2D(512, (1, 1), data_format=IMAGE_ORDERING, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = resize_image(x, strides, data_format=IMAGE_ORDERING)

    return x


def _pspnet(n_classes, encoder, input_height=384, input_width=384):
    assert input_height % 192 == 0
    assert input_width % 192 == 0

    img_input, levels = encoder(input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    # 对f5进行不同程度的池化
    pool_factors = [1, 2, 3, 6]
    pool_outs = [o]

    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)

    # 连接
    o = Concatenate(axis=MERGE_AXIS)(pool_outs)

    # 卷积
    o = Conv2D(512, (1, 1), data_format=IMAGE_ORDERING, use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    # 此时输出为[144,144,nclasses]
    o = Conv2D(n_classes, (3, 3), data_format=IMAGE_ORDERING, padding='same')(o)
    o = resize_image(o, (8, 8), data_format=IMAGE_ORDERING)

    o = UpSampling2D(size=(4,4))(o)




    o = Activation('sigmoid')(o)

    model = Model(img_input, o)
    return model


def mobilenet_pspnet(nclasses, input_height=384, input_width=384):
    model = _pspnet(nclasses, get_mobilenet_encoder, input_height=input_height, input_width=input_width)
    model.model_name = "mobilenet_pspnet"
    return model
