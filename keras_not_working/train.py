import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.callbacks import ModelCheckpoint
from soclevr import load_all, Timer
import os
import argparse
import numpy as np
from model import RN, RN2


def run(attempt, gpunum):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpunum
    def get_session(gpu_fraction=1):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                    allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    ktf.set_session(get_session())


    with Timer("load_all..."):
        rel_train, rel_test, norel_train, norel_test = load_all(source='shm')
        # rel_train, rel_test, norel_train, norel_test = load_all(source='file')

    # model = RN()
    model = RN2()

    model.fit([rel_train[0], rel_train[1]], rel_train[2], validation_data=[[rel_test[0], rel_test[1]], rel_test[2]],
              epochs=100, batch_size=64)


    # with Timer("Build model..."):
    #     input_shape = (maxlen,)
    #     model_input = Input(shape=input_shape)
    #     model = CNNv1(model_input, max_features, embedding)
    #     # model = CNNv2(model_input, max_features)

    # # checkpoint
    # filepath='./model/best-%d-%d' % (w2vdim, attempt)
    #
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    #
    # model.fit(x_trn, y_trn,
    #           batch_size=batch_size,
    #           shuffle=True,
    #           callbacks=callbacks_list,
    #           epochs=epochs,
    #           validation_data=(x_dev, y_dev))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default=0, choices=range(10), type=int)
    parser.add_argument('-g', default="0", choices=["0", "1", "2", "3"], type=str)
    args = parser.parse_args()

    run(args.t, args.g)