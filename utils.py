import math
import torch
import pickle
import numpy as np
import scipy.io as sio
from numpy.random import randint
from numpy.random import shuffle
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder


class IncompleteMultiViewDataset(Dataset):
    def __init__(self, dataset, mask):
        #X is a list of 2d ndarray with the shape of [N,D], Y is a 1d ndarray
        #mask is the shape of[N, V]
        super().__init__()
        self.X = list(zip(*dataset[0]))
        self.Y = dataset[1].astype(np.long)
        self.mask = mask
        self.view_dims = [x.shape[1] for x in dataset[0]]
        self.view_num  = len(self.view_dims)
        self.statistic = [np.mean(dataset[0][i][mask[:,i].astype(np.bool)],axis=0) for i in range(self.view_num)]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        m = self.mask[index]
        return x, y, m


class Bank(object):
    def __init__(self, X, mask):
        super().__init__()
        self.X = list(map(lambda x:torch.from_numpy(x),X))
        self.mask = torch.from_numpy(mask).to(dtype=torch.bool)
        self.sample_num, self.view_num = mask.shape
        self.indexes = torch.arange(self.sample_num).to(dtype=torch.long)
        self.valid = [self.indexes[mask[:,i]] for i in range(self.view_num)]

    def sample(self,size):
        return [self.X[i][self.valid[i][torch.randint(len(self.valid[i]),(size,)).to(dtype=torch.long)]] for i in range(self.view_num)]


def Normalize(data):
    """
    :param data:Input data
    :return:normalized data
    """
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)


def read_data(str_name, ratio, Normal=1):
    """read data and spilt it train set and test set evenly
    :param str_name:path and dataname
    :param ratio:training set ratio
    :param Normal:do you want normalize
    :return:dataset and view number
    """
    if str_name == 'Kinetics-Sounds':
        train_data = pickle.load(open('/home/qsw/LHGN_TMM_2022-main/KS_train.pkl','rb'))
        test_data  = pickle.load(open('/home/qsw/LHGN_TMM_2022-main/KS_test.pkl','rb'))

        audio_train = np.stack(train_data['A'],axis=0)
        video_train = np.stack(train_data['V'],axis=0)
        audio_test  = np.stack(test_data['A'],axis=0)
        video_test  = np.stack(test_data['V'],axis=0)

        if Normal:
            audio_train = Normalize(audio_train)
            video_train = Normalize(video_train)
            audio_test  = Normalize(audio_test)
            video_test  = Normalize(video_test)


        '''
        audio_train = torch.tensor(audio_train)
        video_train = torch.tensor(video_train)
        audio_test  = torch.tensor(audio_test)
        video_test  = torch.tensor(video_test)
        '''

        X_train = [audio_train, video_train]
        X_test  = [audio_test,  video_test]

        labels_train = np.array(train_data['Y'])#torch.tensor(train_data['Y']).unsqueeze(1)
        labels_test  = np.array(test_data['Y'])#torch.tensor(test_data['Y']).unsqueeze(1)
        assert type(labels_train) == np.ndarray and len(labels_train.shape) == 1
        assert type(labels_test) == np.ndarray and len(labels_test.shape) == 1
        traindata = X_train, labels_train
        testdata  = X_test,  labels_test
        return traindata, testdata





    data = sio.loadmat(str_name)
    view_number = data['X'].shape[1]
    X = np.split(data['X'], view_number, axis=1)
    X_train = []
    X_test = []
    labels_train = []
    labels_test = []
    if min(data['gt']) == 0:
        labels = data['gt'] + 1
    else:
        labels = data['gt']
    classes = max(labels)[0]
    all_length = 0
    for c_num in range(1, classes + 1):
        c_length = np.sum(labels == c_num)
        index = np.arange(c_length)
        shuffle(index)
        labels_train.extend(labels[all_length + index][0:math.floor(c_length * ratio)])
        labels_test.extend(labels[all_length + index][math.floor(c_length * ratio):])
        X_train_temp = []
        X_test_temp = []
        for v_num in range(view_number):
            X_train_temp.append(X[v_num][0][0].transpose()[all_length + index][0:math.floor(c_length * ratio)])
            X_test_temp.append(X[v_num][0][0].transpose()[all_length + index][math.floor(c_length * ratio):])
        if c_num == 1:
            X_train = X_train_temp
            X_test = X_test_temp
        else:
            for v_num in range(view_number):
                X_train[v_num] = np.r_[X_train[v_num], X_train_temp[v_num]]
                X_test[v_num] = np.r_[X_test[v_num], X_test_temp[v_num]]
        all_length = all_length + c_length
    if (Normal == 1):
        for v_num in range(view_number):
            X_train[v_num] = Normalize(X_train[v_num])
            X_test[v_num] = Normalize(X_test[v_num])

    
    assert sum([len(e.shape) == 2 for e in X_train]) == len(X_train)
    assert sum([len(e.shape) == 2 for e in X_test]) == len(X_test)
    labels_train = np.array(labels_train).squeeze()
    labels_test  = np.array(labels_test).squeeze() 
    assert type(labels_train) == np.ndarray and len(labels_train.shape) == 1
    assert type(labels_test) == np.ndarray and len(labels_test.shape) == 1
    
    traindata = X_train, labels_train
    testdata  = X_test,  labels_test
    return traindata, testdata


def prepare_data(str_name, ratio, Normal=1):
    """read data and spilt it train set and test set evenly
    :param str_name:path and dataname
    :param ratio:training set ratio
    :param Normal:do you want normalize
    :return:dataset and view number
    """
    data = sio.loadmat(str_name)
    view_number = data['X'].shape[1]
    X = np.split(data['X'], view_number, axis=1)
    X_train = []
    X_valid = []
    X_test = []
    labels_train = []
    labels_valid = []
    labels_test = []
    if min(data['gt']) == 0:
        labels = data['gt'] + 1
    else:
        labels = data['gt']
    classes = max(labels)[0]
    all_length = 0
    for c_num in range(1, classes + 1):
        c_length = np.sum(labels == c_num)
        index = np.arange(c_length)
        shuffle(index)
        labels_train.extend(labels[all_length + index][0:math.floor(c_length * ratio * 0.75)])
        labels_valid.extend(labels[all_length + index][math.floor(c_length * ratio * 0.75):math.floor(c_length * ratio)])#
        labels_test.extend(labels[all_length + index][math.floor(c_length * ratio):])
        X_train_temp = []
        X_valid_temp = []
        X_test_temp = []
        for v_num in range(view_number):
            X_train_temp.append(X[v_num][0][0].transpose()[all_length + index][0:math.floor(c_length * ratio * 0.75)])
            X_valid_temp.append(X[v_num][0][0].transpose()[all_length + index][math.floor(c_length * ratio * 0.75):math.floor(c_length * ratio)])#
            X_test_temp.append(X[v_num][0][0].transpose()[all_length + index][math.floor(c_length * ratio):])
        if c_num == 1:
            X_train = X_train_temp
            X_valid = X_valid_temp
            X_test = X_test_temp
        else:
            for v_num in range(view_number):
                X_train[v_num] = np.r_[X_train[v_num], X_train_temp[v_num]]
                X_valid[v_num] = np.r_[X_valid[v_num], X_valid_temp[v_num]]
                X_test[v_num] = np.r_[X_test[v_num], X_test_temp[v_num]]
        all_length = all_length + c_length
    if (Normal == 1):
        for v_num in range(view_number):
            X_train[v_num] = Normalize(X_train[v_num])
            X_valid[v_num] = Normalize(X_valid[v_num])
            X_test[v_num] = Normalize(X_test[v_num])

    assert all_length == len(labels)
    assert sum([len(e.shape) == 2 for e in X_train]) == len(X_train)
    assert sum([len(e.shape) == 2 for e in X_valid]) == len(X_valid)
    assert sum([len(e.shape) == 2 for e in X_test]) == len(X_test)
    labels_train = np.array(labels_train).squeeze()
    labels_valid = np.array(labels_valid).squeeze()
    labels_test  = np.array(labels_test).squeeze() 
    assert type(labels_train) == np.ndarray and len(labels_train.shape) == 1
    assert type(labels_valid) == np.ndarray and len(labels_valid.shape) == 1
    assert type(labels_test) == np.ndarray and len(labels_test.shape) == 1
    
    traindata = X_train, labels_train
    validdata = X_valid, labels_valid
    testdata  = X_test,  labels_test
    return traindata, validdata, testdata


def get_mask(view_num, data_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data.

        Args:
          view_num: view number
          data_len: number of samples
          missing_rate: Defined in section 4.1 of the paper
        Returns:
          mask

    """
    missing_rate = missing_rate / view_num
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(data_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        one_num = view_num * data_len * one_rate - data_len
        ratio = one_num / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * data_len)
        error = abs(one_rate - ratio)

    return matrix