import numpy as np
from pyshm import SharedNPArray

n_trn = 128
n_tst = 64

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

def load_all():
    rel_train = get_share_memory_tuple('rn_rel_train')
    rel_test = get_share_memory_tuple('rn_rel_test')
    norel_train = get_share_memory_tuple('rn_norel_train')
    norel_test = get_share_memory_tuple('rn_norel_test')

    return rel_train, rel_test, norel_train, norel_test

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
