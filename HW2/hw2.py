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


def normalize_data(data):
    data = [numpy.concatenate(x) for x in data]
    data = numpy.array([zscore(x) for x in data])
    return data


def get_data(t):
    if (t == 'test'):
        images, targets = load_mnist(dataset='testing', path='../HW1/Regression')
        images = normalize_data(images)
        targets = numpy.array([[int(t[0] == i) for i in range(10)] for t in targets])
        return images, targets
    if (t == 'train'):
        images, targets = load_mnist(dataset='training', path='../HW1/Regression')
        images = normalize_data(images)
        targets = numpy.array([[int(t[0] == i) for i in range(10)] for t in targets])
        return images, targets
    raise Exception('You brokeded it...')


def fake_data():
    max_index = lambda x : max([(i,e) for i,e in enumerate(x)],key=lambda y : y[1])[0]
    X = 4
    Y = 1000
    data = [numpy.array([random.random() for x in range(X)]) for y in range(Y)]
    targets = [max_index(x) for x in data]
    targets = numpy.array([[int(t == i) for i in range(X)] for t in targets])
    return data,targets

    
def HLN_tester():
    max_index = lambda x : max([(i,e) for i,e in enumerate(x)],key=lambda y : y[1])[0]
    images, targets = get_data('train')
    #images, targets = fake_data()
    machine = HLN(images, targets,patience=5000,rate=10**-3,batch=50,hidden=[50])
    for w in machine.weights:
        print(w.shape)
    images,targets = get_data('test')
    count = 0
    s = len(images)
    for d,t in zip(images[:s],targets[:s]):
        p = machine.predict(d)
        p = max_index(p[-1])
        if(t[p]==1):
            count+=1
    print('Accuracy : '+str(count/s))


def softmax(vect):
    vect = numpy.exp(vect)
    return vect / sum(vect)


class HLN:
    def __init__(self, data, targets, rate=10**-4, drop=0.5, patience=100000, batch=100, hidden=[100],act_func=softmax):
        norm = lambda scale: numpy.random.normal(scale=scale)
        scale_func = lambda x : 1/numpy.sqrt(x+1)
        layers = [len(data[0])]+list(hidden)+[len(targets[0])]
        self.act_func = act_func
        self.rate = rate/batch
        self.weights = []
        self.bias = []
        for i in range(1,len(layers)):
            self.weights.append(numpy.matrix(
                [[norm(scale_func(layers[i-1])) for x in range(layers[i-1])] for y in range(layers[i])]))
            self.bias.append(numpy.matrix([norm(scale_func(layers[i-1])) for x in range(layers[i])]).transpose())
        self.transpose_weights = [x.transpose() for x in self.weights]
        data = numpy.array([[d, t] for d, t in zip(data, targets)])
        numpy.random.shuffle(data)
        count=0
        matrix_sum_square = lambda m : sum(x**2 for x in numpy.nditer(m))
        for _ in range(patience):
            if (count + batch > len(data)):
                numpy.random.shuffle(data)
                count = 0
            weight_update = [numpy.matrix([[0 for x in y] for y in z]) for z in self.weights]
            bias_update = [numpy.matrix([0 for x in y]).transpose() for y in self.bias]
            for i in range(batch):
                i += count
                prediction = self.predict(data[i][0])
                wu, bu = self.update(prediction, data[i][1])
                weight_update = [w_old + w_new for w_old, w_new in zip(weight_update, wu)]
                bias_update = [b_old + b_new for b_old, b_new in zip(bias_update, bu)]    
            count += batch
            self.weights = [w_old + w_new for w_old, w_new in zip(self.weights, weight_update)]
            self.transpose_weights = [x.transpose() for x in self.weights]
            self.bias = [b_old + b_new for b_old, b_new in zip(self.bias, bias_update)]

    def predict(self, datum, activation=None):
        if (activation == None):
            activation = lambda val : 1.7159 * numpy.tanh(2 / 3 * val)
        outputs = []
        outputs.append(numpy.matrix(datum).transpose())
        count = 0
        for w, b in zip(self.weights, self.bias):
            count+=1
            if(len(self.weights)==count):
                break
            outputs.append(activation(numpy.dot(w,outputs[-1]))+b)
        outputs.append(self.act_func(numpy.dot(self.weights[-1], outputs[-1]) + self.bias[-1]))
        return outputs

    def update(self, out, t):
        deltas = []
        t = numpy.matrix(t).transpose()
        final_delta = t - out[-1]
        deltas.append(final_delta)
        count = 0
        for w, o in zip(reversed(self.transpose_weights), reversed(out[:-1])):
            count+=1
            if(count==len(self.weights)):
                break
            temp = numpy.dot(w,deltas[-1])
            temp = numpy.multiply(temp,o)
            deltas.append(temp)
        update = []
        for d, o in zip(reversed(deltas), out):
            update.append(self.rate * numpy.dot(d, o.transpose()))
        bias_update = [self.rate * d for d in reversed(deltas)]
        return update, bias_update


if (__name__ == '__main__'):
    print('Start Time : '+time.asctime())
    start = time.time()
    HLN_tester()
    stop = time.time()
    print('Execution Time : ' + str(stop - start))
