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


rel_trn, rel_dev, rel_tst, norel_trn, norel_dev, norel_tst = load_data()

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

shm_rel_trn = set_share_memory_tuple(rel_trn, 'rn_rel_trn')
shm_rel_dev = set_share_memory_tuple(rel_dev, 'rn_rel_dev')
shm_rel_tst = set_share_memory_tuple(rel_tst, 'rn_rel_tst')
shm_norel_trn = set_share_memory_tuple(norel_trn, 'rn_norel_trn')
shm_norel_dev = set_share_memory_tuple(norel_dev, 'rn_norel_dev')
shm_norel_tst = set_share_memory_tuple(norel_tst, 'rn_norel_tst')

print rel_tst[2][0:10]

print 'now sleep...'
sys.stdout.flush()


def signal_handler(signal, frame):
    print('now terminated...')
    sys.exit(0)

# use 'kill -2 ###' to delete all /dev/shm/*
signal.signal(signal.SIGINT, signal_handler)
signal.pause()
