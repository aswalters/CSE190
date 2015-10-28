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

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

def normalize_data(data,pca_flag=False):
    data = [numpy.concatenate(x) for x in data]
    data = numpy.array([zscore(x) for x in data])
    if(pca_flag):
        pca = PCA(20)
        data = pca.fit_transform(data)
    return data

def get_data(t):
    if(t=='test'):
        images,targets = load_mnist(dataset='testing',path='../HW1/Regression')
        #images = images[:2000]
        #targets = targets[:2000]
        images = normalize_data(images)
        targets = numpy.array([[int(t[0] == i) for i in range(10)] for t in targets])
        return images,targets
    if(t=='train'):
        images,targets = load_mnist(dataset='training',path='../HW1/Regression')
        #images = images[:20000]
        #targets = targets[:20000]
        images = normalize_data(images)
        targets = numpy.array([[int(t[0] == i) for i in range(10)] for t in targets])
        return images,targets
    raise Exception('You broken it...')

def HLN_tester():
    images,targets = get_data('train')
    HLN(images,targets,hidden=(2,1))

def softmax(vect):
    vect = numpy.exp(vect)
    return vect / sum(vect)

class HLN:

    def __init__(self, data, targets, rate=0.01, drop=0.5, patience=100000, batch=100, hidden=(10,2),act_func=softmax):

        self.act_func = act_func
        self.I = len(data[0])
        self.J, self.L = hidden
        self.K = len(targets[0])
        self.a = rate

        norm = lambda scale : numpy.random.normal(scale=scale)

        scale = 1 / numpy.sqrt(self.I + 1)
        self.initial_weights = numpy.array([[norm(scale) for x in range(self.I)] for y in range(self.J)])
        self.J1_bias = numpy.array([norm(scale) for y in range(self.J)])

        scale = 1 / numpy.sqrt(self.J + 1)
        self.hidden_weights = numpy.array([[[norm(scale) for x in range(self.J)] for y in range(self.J)] for z in range(self.L - 1)])
        self.JJ_bias = numpy.array([[norm(scale) for x in range(self.J)] for y in range(self.L-1)])
        self.final_weights = numpy.array([[norm(scale) for x in range(self.J)] for y in range(self.K)])
        self.K_bias = numpy.array([norm(scale) for x in range(self.K)])

        for _ in range(patience):
            start = random.randint(0, self.I)
            end = start + batch
            for image in data[start:end]:
                out = self.predict(image)

    def predict(self, datum):
        activation = lambda val : 1.7159*numpy.tanh(2/3*val)
        outputs = []
        outputs.append(activation(numpy.dot(self.initial_weights, datum) + self.J1_bias))
        for weights,bias in zip(self.hidden_weights,self.JJ_bias):
            outputs.append(activation(numpy.dot(weights, outputs[-1]) + bias))
        outputs.append(self.act_func(numpy.dot(self.final_weights, outputs[-1]) + self.K_bias))
        return outputs

    def update(self, out,t):
        final_delta = t-out[-1]
        hidden_deltas = []
        hidden_deltas.append([out[-2][j]*(numpy.dot(j,final_delta)+self.JJ_bias[-1][j]) for j in self.final_weights])
        for i in range(self.L-2,-1,-1):
            hidden_deltas.append([out[i][j]*(numpy.dot(self.hidden_weights[i][j],hidden_deltas[-1])self.JJ_bias) for j in range(self.J)])


        hidden_deltas = [[ for j in w] for w in self.hidden_weights]
        final = [[ self.a*delta[k]*out[-2][j] for j,w in enumerate(vect)] for k,vect in self.final_weights]



        final = numpy.array([w_vect + self.a*(final_delta)*out[-2] for w_vect in self.final_weights])
        #final = numpy.array([w_vect + self.a*(t-out[-1])*out[-2] for w_vect in self.final_weights])






if(__name__=='__main__'):
    start = time.time()
    HLN_tester()
    stop = time.time()
    print('Execution Time : '+str(stop-start))



