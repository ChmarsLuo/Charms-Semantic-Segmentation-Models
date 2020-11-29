from keras.models import *
from keras.layers import *
import keras.backend as K
import keras

IMAGE_ORDERING = 'channels_last'

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



#################################################
#          segnet_monbilenet                    #
#################################################

IMAGE_ORDERING = 'channels_last'
def segnet_decoder(f, n_classes, n_up=3):
    assert n_up >= 2

    o = f
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    # 进行一次UpSampling2D，此时hw变为原来的1/8
    # 52,52,512
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    # 进行一次UpSampling2D，此时hw变为原来的1/4
    # 104,104,256
    for _ in range(n_up - 2):
        o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
        o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
        o = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)

    # 进行一次UpSampling2D，此时hw变为原来的1/2
    # 208,208,128
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    # 此时输出为h_input/2,w_input/2,nclasses
    o = Conv2D(n_classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)

    return o


def _segnet(n_classes, encoder, input_height=416, input_width=608, encoder_level=3):
    # encoder通过主干网络
    img_input, levels = encoder(input_height=input_height, input_width=input_width)

    # 获取hw压缩四次后的结果
    feat = levels[encoder_level]

    # 将特征传入segnet网络
    o = segnet_decoder(feat, n_classes, n_up=3)

    # 将结果进行reshape
    o = UpSampling2D(size=(2,2))(o)
    o = Activation('sigmoid')(o)
    model = Model(img_input, o)

    return model


def mobilenet_segnet(nclasses, input_height=224, input_width=224, encoder_level=3):
    model = _segnet(nclasses, get_mobilenet_encoder, input_height=input_height, input_width=input_width,
                    encoder_level=encoder_level)
    model.model_name = "mobilenet_segnet"
    return model

