import keras.backend as K
from keras.models import Model


def scheduler(epoch, model):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 1 == 0 and epoch != 0:
        # 以Numpy array的形式返回张量的值
        lr = K.get_value(model.optimizer.lr)
        # 从numpy array将值载入张量中
        K.set_value(model.optimizer.lr, lr * 0.95)
        print("lr changed to {}".format(lr * 0.95))
    return K.get_value(model.optimizer.lr)