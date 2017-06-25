import argparse
import os
import cPickle as pickle
import random
import numpy as np
import math
import sys

from soclevr import load_all, DataFeeder, Timer
from model import RN
import tensorflow as tf

BATCH_SIZE = 512

# from tensorflow.contrib.keras.python.keras.layers import BatchNormalization
BatchNormalization = tf.contrib.keras.layers.BatchNormalization
Dense = tf.contrib.keras.layers.Dense
Dropout = tf.contrib.keras.layers.Dropout

# from tensorflow.contrib.layers.python.layers import batch_norm

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

with Timer('Loading data...'):
    rel_trn, rel_dev, rel_tst, norel_trn, norel_dev, norel_tst = load_all(source='shm')
    trn_feader = DataFeeder(norel_trn, BATCH_SIZE)
    dev_feader = DataFeeder(norel_dev, BATCH_SIZE)


with tf.Graph().as_default():
    model = RN()
    # model = RN2()

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    # optimizer = tf.train.AdamOptimizer(1e-3)
    # grads_and_vars = optimizer.compute_gradients(model.loss)
    # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(model.loss)

    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(visible_device_list=str('0'), allow_growth=True)  # o
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False))

    sess.run(init)

    feed_dict = {}
    for i in range(30000): # 30,000steps~20 epoch
        img_batch, question_batch, answer_batch_ = trn_feader.get_next()

        answer_batch = np.zeros((len(answer_batch_), 10))
        answer_batch[np.arange(len(answer_batch_)), answer_batch_] = 1

        feed_dict[model.img] = img_batch
        feed_dict[model.question] = question_batch
        feed_dict[model.answer] = answer_batch

        # _, preds, acc = sess.run([train_op, model.pred, model.accuracy], feed_dict=feed_dict)
        _, loss, acc = sess.run([train_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # print preds
        # print answer_batch
        print 'TRAIN[%d] loss=%f, acc=%f' % (i, loss, acc)
        sys.stdout.flush()

        if i % 50 == 0:
            # if i == 0:
            #     continue

            #evaluation
            acc_list = []
            loss_list = []
            for k in range(int(math.ceil(dev_feader.n*1.0/dev_feader.batch_size))):
                img_dev, question_dev, answer_dev_ = dev_feader.get_next()
                answer_dev = np.zeros((len(answer_dev_), 10))
                answer_dev[np.arange(len(answer_dev_)), answer_dev_] = 1

                feed_dict[model.img] = img_dev
                feed_dict[model.question] = question_dev
                feed_dict[model.answer] = answer_dev

                loss, acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
                acc_list.append(acc)
                loss_list.append(loss)

            print 'DEV[%d] loss=%f, acc=%f' % (i, np.mean(loss_list), np.mean(acc_list))
            sys.stdout.flush()

print 'done'

