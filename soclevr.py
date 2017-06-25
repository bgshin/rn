import numpy as np
from pyshm import SharedNPArray
import time
import os
import pickle
import random
import math

# # samples
# n_trn = 128
# n_tst = 64

# real dataset
n_trn = 9800
n_tst = 200

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)


def get_type(idx):
    type_dic = {0: np.float64, 1: np.float64, 2: np.int64}
    return type_dic[idx]


def get_shape(idx, sample_size):
    shape_dic = {0: (sample_size*10, 3, 75, 75), 1: (sample_size*10, 11), 2: (sample_size*10,)}
    return shape_dic[idx]


def get_sample_size(name):
    if 'train' in name:
        return n_trn
    else:
        return n_tst



def get_share_memory_tuple(name):
    data_list = []
    for i in range(3):
        data =SharedNPArray(shape=get_shape(i, get_sample_size(name)), dtype=get_type(i), tag=name+'_%d' % i, create=False)
        data_list.append(data)

    return tuple(data_list)

def load_all(source):
    if source=='shm':
        rel_trn = get_share_memory_tuple('rn_rel_trn')
        rel_dev = get_share_memory_tuple('rn_rel_dev')
        rel_tst = get_share_memory_tuple('rn_rel_tst')

        norel_trn = get_share_memory_tuple('rn_norel_trn')
        norel_dev = get_share_memory_tuple('rn_norel_dev')
        norel_tst = get_share_memory_tuple('rn_norel_tst')
        return rel_trn, rel_dev, rel_tst, norel_trn, norel_dev, norel_tst

    else:
        print('loading data...')
        dirs = './data'
        # filename = os.path.join(dirs, 'sort-of-clevr-sample.pickle')
        filename = os.path.join(dirs, 'sort-of-clevr.pickle')
        f = open(filename, 'r')
        trn_datasets, dev_datasets, tst_datasets = pickle.load(f)
        rel_trn = []
        rel_dev = []
        rel_tst = []

        norel_trn = []
        norel_dev = []
        norel_tst = []

        for img, relations, norelations in trn_datasets:
            img = np.swapaxes(img, 0, 2)
            for qst, ans in zip(relations[0], relations[1]):
                rel_trn.append((img, qst, ans))
            for qst, ans in zip(norelations[0], norelations[1]):
                norel_trn.append((img, qst, ans))

        for img, relations, norelations in dev_datasets:
            img = np.swapaxes(img, 0, 2)
            for qst, ans in zip(relations[0], relations[1]):
                rel_dev.append((img, qst, ans))
            for qst, ans in zip(norelations[0], norelations[1]):
                norel_dev.append((img, qst, ans))

        for img, relations, norelations in tst_datasets:
            img = np.swapaxes(img, 0, 2)
            for qst, ans in zip(relations[0], relations[1]):
                rel_tst.append((img, qst, ans))
            for qst, ans in zip(norelations[0], norelations[1]):
                norel_tst.append((img, qst, ans))

        print('converting data...')
        datasets = [rel_trn, rel_dev, rel_tst, norel_trn, norel_dev, norel_tst]
        random.seed(42)
        for dataset in datasets:
            random.shuffle(dataset)
        n_datasets = []
        for dataset in datasets:
            img = [e[0] for e in dataset]
            qst = [e[1] for e in dataset]
            ans = [e[2] for e in dataset]
            n_datasets.append((img, qst, ans))

        return tuple(n_datasets)


class DataFeeder(object):
    # __metaclass__ = Singleton

    def __init__(self, trn_data, batch_size=None, shuffle=True):
        self.img = np.swapaxes(np.array(trn_data[0]), 1, 3)
        self.question = np.array(trn_data[1])
        self.answer = np.array(trn_data[2])

        self.n = len(self.img)
        self.shuffle = shuffle

        if batch_size == None:
            self.batch_size = self.n
        else:
            self.batch_size = min(self.n, batch_size)

        self.set_batch(self.batch_size)

    def set_batch(self, batch_size=0):
        self.batch_size = min(self.n, batch_size)

        if self.batch_size == 0:  # all samples
            self.num_batches_per_epoch = self.n

        else:
            self.num_batches_per_epoch = math.ceil(self.n * 1.0 / self.batch_size) + 1

        np.random.seed(3)  # FIX RANDOM SEED
        self.batch_init()

    def batch_init(self):
        if self.shuffle:
            self.shuffle_indices = np.random.permutation(np.arange(self.n))

        else:
            self.shuffle_indices = np.arange(self.n)

        self.shuffled_img = self.img[self.shuffle_indices]
        self.shuffled_question = self.question[self.shuffle_indices]
        self.shuffled_answer = self.answer[self.shuffle_indices]

        self.batch_num = 0
        self.start_index = self.batch_num * self.batch_size
        self.end_index = min((self.batch_num + 1) * self.batch_size, self.n)

    def set_batch_all(self):
        self.set_batch(self.n)

    def get_sample(self):
        return self.img[0], self.question[0], self.answer[0]

    def get_next(self):
        index_log = '%d:%d' % (self.start_index, self.end_index)
        img_batch = self.shuffled_img[self.start_index:self.end_index]
        question_batch = self.shuffled_question[self.start_index:self.end_index]
        answer_batch = self.shuffled_answer[self.start_index:self.end_index]
        self.batch_num += 1

        if self.batch_num * self.batch_size >= self.n:
            self.batch_init()

        else:
            self.start_index = self.batch_num * self.batch_size
            self.end_index = min((self.batch_num + 1) * self.batch_size, self.n)

        return img_batch, question_batch, answer_batch

# def load_data():
#     print('loading data...')
#     dirs = './data'
#     filename = os.path.join(dirs, 'sort-of-clevr-sample.pickle')
#     # filename = os.path.join(dirs, 'sort-of-clevr.pickle')
#     f = open(filename, 'r')
#     train_datasets, test_datasets = pickle.load(f)
#     rel_train = []
#     rel_test = []
#     norel_train = []
#     norel_test = []
#     for img, relations, norelations in train_datasets:
#         img = np.swapaxes(img, 0, 2)
#         for qst, ans in zip(relations[0], relations[1]):
#             rel_train.append((img, qst, ans))
#         for qst, ans in zip(norelations[0], norelations[1]):
#             norel_train.append((img, qst, ans))
#
#     for img, relations, norelations in test_datasets:
#         img = np.swapaxes(img, 0, 2)
#         for qst, ans in zip(relations[0], relations[1]):
#             rel_test.append((img, qst, ans))
#         for qst, ans in zip(norelations[0], norelations[1]):
#             norel_test.append((img, qst, ans))
#
#     print('converting data...')
#     datasets = [rel_train, rel_test, norel_train, norel_test]
#     random.seed(42)
#     for dataset in datasets:
#         random.shuffle(dataset)
#     n_datasets = []
#     for dataset in datasets:
#         img = [e[0] for e in dataset]
#         qst = [e[1] for e in dataset]
#         ans = [e[2] for e in dataset]
#         n_datasets.append((img, qst, ans))
#
#     return tuple(n_datasets)
#
# def np_checker(a,b):
#     for i in range(3):
#         if np.array_equal(a[i], b[i]) is False:
#             return False
#
#     return True
#
# rel_train, rel_test, norel_train, norel_test = load_data()
#
# rel_train2, rel_test2, norel_train2, norel_test2= load_all()
#
#
# print np_checker(rel_train, rel_train2)
# print np_checker(rel_test, rel_test2)
# print np_checker(norel_train, norel_train2)
# print np_checker(norel_test, norel_test2)
#
#
#
# print 'd'
