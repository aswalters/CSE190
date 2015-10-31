from __future__ import division
import numpy
import random
import time
from sklearn.decomposition import PCA
from scipy.stats import zscore
import os
import struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
from itertools import chain


def load_mnist(dataset="training", digits=numpy.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


def normalize_data(data, pca_flag=False):
    data = [numpy.concatenate(x) for x in data]
    data = numpy.array([zscore(x) for x in data])
    if (pca_flag):
        pca = PCA(20)
        data = pca.fit_transform(data)
    return data


def get_data(t,pca_flag=False):
    if (t == 'test'):
        images, targets = load_mnist(dataset='testing', path='../HW1/Regression')
        # images = images[:2000]
        # targets = targets[:2000]
        images = normalize_data(images,pca_flag)
        targets = numpy.array([[int(t[0] == i) for i in range(10)] for t in targets])
        return images, targets
    if (t == 'train'):
        images, targets = load_mnist(dataset='training', path='../HW1/Regression')
        # images = images[:20000]
        # targets = targets[:20000]
        images = normalize_data(images,pca_flag)
        targets = numpy.array([[int(t[0] == i) for i in range(10)] for t in targets])
        return images, targets
    raise Exception('You broken it...')


def HLN_tester():
    images, targets = get_data('train')
    machine = HLN(images, targets, hidden=(100, 1),patience=1000)
    images,targets = get_data('test')
    max_index = lambda x : max([(i,e) for i,e in enumerate(x)],key=lambda y : y[1])[0]
    count = 0
    for d,t in zip(images,targets):
        p = machine.predict(d)
        p = max_index(p[-1])
        if(t[p]!=1):
            count+=1
    print(count/len(images))


def softmax(vect):
    vect = numpy.exp(vect)
    return vect / sum(vect)


class HLN:
    def __init__(self, data, targets, rate=0.01, drop=0.5, patience=100000, batch=100, hidden=(10, 2),
                 act_func=softmax):

        self.act_func = act_func
        self.I = len(data[0])
        self.J, self.L = hidden
        self.K = len(targets[0])
        self.a = rate
        self.weights = []
        self.bias = []

        norm = lambda scale: numpy.random.normal(scale=scale)

        scale = 1 / numpy.sqrt(self.I + 1)
        initial_weights = numpy.array([[norm(scale) for x in range(self.I)] for y in range(self.J)])
        J1_bias = numpy.array([norm(scale) for y in range(self.J)])

        scale = 1 / numpy.sqrt(self.J + 1)
        hidden_weights = numpy.array(
            [[[norm(scale) for x in range(self.J)] for y in range(self.J)] for z in range(self.L - 1)])
        JJ_bias = numpy.array([[norm(scale) for x in range(self.J)] for y in range(self.L - 1)])
        final_weights = numpy.array([[norm(scale) for x in range(self.J)] for y in range(self.K)])
        K_bias = numpy.array([norm(scale) for x in range(self.K)])

        self.weights.append(initial_weights)
        for w in hidden_weights:
            self.weights.append(w)
        self.weights.append(final_weights)

        self.bias.append(J1_bias)
        for b in JJ_bias:
            self.bias.append(b)
        self.bias.append(K_bias)

        data = numpy.array([[d, t] for d, t in zip(data, targets)])
        numpy.random.shuffle(data)
        count = 0
        TRACK = 0
        for _ in range(patience):

            weight_update = [numpy.array([[0 for x in y] for y in z]) for z in self.weights]
            bias_update = [numpy.array([0 for x in y]) for y in self.bias]
            if (count + batch > len(data)):
                numpy.random.shuffle(data)
                count = 0
            for i in range(batch):
                i = count + i
                if TRACK == 1:
                    print("****")
                    for b in self.bias:
                        print(b.shape)
                    print('****')
                    print(i)
                    print(len(data[i][0]))
                    print('****')
                prediction = self.predict(data[i][0])
                wu, bu = self.update(prediction, data[i][1])

                weight_update = [w_old + w_new for w_old, w_new in zip(weight_update, wu)]
                bias_update = [b_old + b_new for b_old, b_new in zip(bias_update, bu)]

            count += batch

            self.weights = [w_old - w_new for w_old, w_new in zip(self.weights, weight_update)]
            self.bias = [b_old - b_new for b_old, b_new in zip(self.bias, bias_update)]

            TRACK += 1



    def predict(self, datum, activation=None):
        if (activation == None):
            activation = lambda val: 1.7159 * numpy.tanh(2 / 3 * val)
        outputs = []
        outputs.append(datum)
        for w, b in zip(self.weights, self.bias):
            print('w : '+str(w.shape))
            print('i : '+str(outputs[-1].shape))
            outputs.append(activation(numpy.dot(w, outputs[-1]) + b))
        outputs[-1] = self.act_func(numpy.dot(self.weights[-1], outputs[-2]) + self.bias[-1])
        return outputs

    def update(self, out, t):
        deltas = []
        final_delta = t - out[-1]
        deltas.append(final_delta)
        for w, o in zip(reversed(self.weights), reversed(out[:-1])):
            deltas.append(numpy.dot(w.transpose(), deltas[-1]) * o)
        deltas = deltas[:-1]
        update = []
        for w, d, o in zip(self.weights, reversed(deltas), out):
            d = numpy.matrix(d)
            o = numpy.matrix(o)
            update.append(self.a * numpy.dot(d.transpose(), o))
        bias_update = [self.a * d for d in reversed(deltas)]

        return update, bias_update


if (__name__ == '__main__'):
    start = time.time()
    HLN_tester()
    stop = time.time()
    print('Execution Time : ' + str(stop - start))
