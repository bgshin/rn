import numpy as np
import sys
import signal
from pyshm import SharedNPArray
import os
import pickle
import random

def load_data():
    print('loading data...')
    dirs = './data'
    filename = os.path.join(dirs, 'sort-of-clevr-sample.pickle')
    # filename = os.path.join(dirs, 'sort-of-clevr.pickle')
    f = open(filename, 'r')
    train_datasets, test_datasets = pickle.load(f)
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    for img, relations, norelations in train_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst, ans in zip(relations[0], relations[1]):
            rel_train.append((img, qst, ans))
        for qst, ans in zip(norelations[0], norelations[1]):
            norel_train.append((img, qst, ans))

    for img, relations, norelations in test_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst, ans in zip(relations[0], relations[1]):
            rel_test.append((img, qst, ans))
        for qst, ans in zip(norelations[0], norelations[1]):
            norel_test.append((img, qst, ans))

    print('converting data...')
    datasets = [rel_train, rel_test, norel_train, norel_test]
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


rel_train, rel_test, norel_train, norel_test = load_data()

def set_share_memory(nparray, name, i):
    nparray = np.array(nparray)
    name = name % i
    print name, nparray.shape, nparray.dtype
    sys.stdout.flush()
    shm_obj = SharedNPArray(shape=nparray.shape, dtype=nparray.dtype, tag=name)
    shm_obj.copyto(nparray)
    return shm_obj

shmobj_list = []
def set_share_memory_tuple(tuples, name):
    for i in range(3):
        shmobj_list.append(set_share_memory(tuples[i], name+'_%d', i))

shm_rel_train = set_share_memory_tuple(rel_train, 'rn_rel_train')
shm_rel_test = set_share_memory_tuple(rel_test, 'rn_rel_test')
shm_norel_train = set_share_memory_tuple(norel_train, 'rn_norel_train')
shm_norel_test = set_share_memory_tuple(norel_test, 'rn_norel_test')
print rel_test[2][0:10]

print 'now sleep...'
sys.stdout.flush()


def signal_handler(signal, frame):
    print('now terminated...')
    sys.exit(0)

# use 'kill -2 ###' to delete all /dev/shm/*
signal.signal(signal.SIGINT, signal_handler)
signal.pause()
