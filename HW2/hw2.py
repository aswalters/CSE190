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
		images,targets = load_mnist(dataset='testing',path='hw1_data')
		images = images[:2000]
		targets = targets[:2000]
		images = normalize_data(images)
		return images,targets
	if(t=='train'):
		images,targets = load_mnist(dataset='training',path='hw1_data')
		images = images[:20000]
		targets = targets[:20000]
		images = normalize_data(images)
		return images,targets
	raise Exception('You broken it...')

def HLN_tester():
	images,targets = get_data('train')
	print(images[0])

class HLN:

	def __init__(self,data,targets,rate=0.01,drop=0.5,patience=100000,batch=100,hidden=(10,1) ):
		pass
		'''
		self.hidden_weights = numpy.array([[numpy.random.normal(scale=1/) for x in hidden[0]] for y in hidden[1]])
		'''

	def predict(self,datum):
		pass


if(__name__=='__main__'):
	start = time.time()
	HLN_tester()
	stop = time.time()
	print('Execution Time : '+str(stop-start))



