# coding=utf-8
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.backend.common import normalize_data_format


class BilinearUpsampling(Layer):

    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
                 input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0], height, width, input_shape[3])

    def call(self, inputs):
        # .tf
        return tf.image.resize_bilinear(inputs, (int(inputs.shape[1] * self.upsampling[0]),
                                                 int(inputs.shape[2] * self.upsampling[1])))

    def get_config(self):

        config = {'size': self.upsampling, 'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def DeepLabV2(nclasses, input_height=256, input_width=256):
    inputs = Input(shape=(input_height, input_width, 3))
    
     # Block 1
    x = ZeroPadding2D(padding=(1, 1))(inputs)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv1_1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv1_2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Block 2
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name='conv2_1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name='conv2_2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Block 3
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', name='conv3_1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', name='conv3_2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', name='conv3_3')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Block 4
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', name='conv4_1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', name='conv4_2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', name='conv4_3')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)

    # Block 5 
    x = ZeroPadding2D(padding=(2, 2))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_1')(x)
    x = ZeroPadding2D(padding=(2, 2))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_2')(x)
    x = ZeroPadding2D(padding=(2, 2))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_3')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    p5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)

    # branching for Atrous Spatial Pyramid Pooling - Until here -14 layers
    # hole = 6
    b1 = ZeroPadding2D(padding=(6, 6))(p5)
    b1 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(6, 6), activation='relu', name='fc6_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Conv2D(filters=nclasses, kernel_size=(1, 1), activation='relu', name='fc8_1')(b1)

    # hole = 12
    b2 = ZeroPadding2D(padding=(12, 12))(p5)
    b2 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(12, 12), activation='relu', name='fc6_2')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_2')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Conv2D(filters=nclasses, kernel_size=(1, 1), activation='relu', name='fc8_2')(b2)

    # hole = 18
    b3 = ZeroPadding2D(padding=(18, 18))(p5)
    b3 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(18, 18), activation='relu', name='fc6_3')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_3')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Conv2D(filters=nclasses, kernel_size=(1, 1), activation='relu', name='fc8_3')(b3)

    # hole = 24
    b4 = ZeroPadding2D(padding=(24, 24))(p5)
    b4 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(24, 24), activation='relu', name='fc6_4')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_4')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Conv2D(filters=nclasses, kernel_size=(1, 1), activation='relu', name='fc8_4')(b4)
    print(b4.shape)

    s = Add()([b1, b2, b3, b4])
    # logits = BilinearUpsampling(upsampling=8)(s)
    logits = UpSampling2D(size=(8,8))(s)

    out = Activation('sigmoid')(logits)

    model = Model(input=inputs, output=out)



    return model
