from data import *
from keras.models import load_model
import os
import keras
from keras.optimizers import *
from lr.lr import scheduler
from keras.losses import binary_crossentropy, categorical_crossentropy
from models import MobileNext
from keras.callbacks import ModelCheckpoint, TensorBoard,\
     LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, CSVLogger



########################################################################################################################
if __name__ == '__main__':
    batch_size = 1
    epochs = 30
    # prepare model
    model = MobileNext()
    model.summary()

    # load weights
    BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                        'releases/download/v0.6/')
    model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ('1_0', 224)
    weight_path = BASE_WEIGHT_PATH +'\\'+ model_name
    weights_path = keras.utils.get_file(model_name, weight_path)
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    # train and val path
    train_im_path,train_mask_path = './data/build/train/images/','./data/build/train/labels/'
    val_im_path,val_mask_path = './data/build/val/images/','./data/build/val/labels/'
    train_set = os.listdir(train_im_path)
    val_set = os.listdir(val_im_path)
    train_number = len(train_set)
    val_number = len(val_set)
    train_root = './data/build/train/'
    val_root = './data/build/val/'

    # data aug and generator
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
    training_generator = trainGenerator(batch_size,train_root,'images','labels',data_gen_args,
                                        image_save_data_dir = None,
                                        mask_save_data_dir = None
                                        )
    validation_generator = trainGenerator(batch_size,val_root,'images','labels',data_gen_args,
                                          image_save_data_dir=None,
                                          mask_save_data_dir=None,
                                          )


    model_path ="./logs/"
    model_name = 'build_{epoch:03d}.h5'
    model_file = os.path.join(model_path, model_name)
    model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.5625), cooldown=0, patience=5, min_lr=0.5e-6)
    model.compile(loss=binary_crossentropy,
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    callable = [EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto'),
                model_checkpoint,
                lr_reducer,
                CSVLogger(filename='./logs/log.csv', append=False),  # CSVLoggerb保存训练结果   好好用
                TensorBoard(log_dir='./logs/')]

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=train_number//batch_size,
                        validation_steps=val_number//batch_size,
                        use_multiprocessing=False,
                        epochs=epochs,verbose=1,
                        initial_epoch = 0,
                        callbacks=callable)
