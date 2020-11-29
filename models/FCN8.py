#coding=utf-8

from keras.applications import vgg16
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv2DTranspose, Input, Cropping2D, add, Dropout, Reshape, Activation


def FCN8_helper(nclasses, input_height=224, input_width=224):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    model = vgg16.VGG16(nclasses)
    assert isinstance(model, Model)

    o = Conv2D(
        filters=4096,
        kernel_size=(
            7,
            7),
        padding="same",
        activation="relu",
        name="fc6")(
            model.output)
    o = Dropout(rate=0.5)(o)
    o = Conv2D(
        filters=4096,
        kernel_size=(
            1,
            1),
        padding="same",
        activation="relu",
        name="fc7")(o)
    o = Dropout(rate=0.5)(o)

    o = Conv2D(filters=nclasses, kernel_size=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal",
               name="score_fr")(o)

    o = Conv2DTranspose(filters=nclasses, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                        name="score2")(o)

    fcn8 = Model(inputs=img_input, outputs=o)
    # mymodel.summary()
    return fcn8



def FCN8(nclasses, input_height=224, input_width=224):


    fcn8 = FCN8_helper(nclasses, input_height, input_width)



	# Conv to be applied on Pool4
    skip_con1 = Conv2D(nclasses, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                       name="score_pool4")(fcn8.get_layer("block4_pool").output)
    Summed = add(inputs=[skip_con1, fcn8.output])

    x = Conv2DTranspose(nclasses, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                        name="score4")(Summed)

    ###
    skip_con2 = Conv2D(nclasses, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                       name="score_pool3")(fcn8.get_layer("block3_pool").output)
    Summed2 = add(inputs=[skip_con2, x])

    #####
    Up = Conv2DTranspose(nclasses, kernel_size=(8, 8), strides=(8, 8),
                         padding="valid", activation=None, name="upsample")(Summed2)

    

    Up = Activation("sigmoid")(Up)

    model = Model(inputs=fcn8.input, outputs=Up)



    return model



	
