import os, struct
import numpy as np
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
from scipy import stats
import math
import random
import matplotlib.pyplot as plt


def loadMNIST(dataset="training", digits=np.arange(10), path="."):
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

#Lambda function to only grab first 20000 training set
trainingData = lambda : (x[0:20000] for x in loadMNIST())
#Lambda function to only grab first 2000 testing set
testingData = lambda : (x[0:2000] for x in loadMNIST('testing'))

def setupData(dataType):
    """
    Computes the Z-Scores for the images and adds the intercept term
    :param dataType: either training or testing
    :return: images, labels
    """
    if dataType != 'training' and dataType != 'testing':
        raise ValueError('dataType must be training or testing')
    if dataType == 'training':
        images, labels = trainingData()
    else:
        images, labels = testingData()
    images = [np.insert(stats.zscore(np.concatenate(x)), 0, 1.0) for x in images]
    return images, labels

class logisticRegression:

    def __init__(self, learn=1, iterations=100, subset=1000):
        """
        Creates a logistic regression object and performs a gradient descent
        :param self: logisticRegression object
        :param learn: learning rate factor, defaults=1
        :param iterations: number of weight training iterations, default=100
        :param subset: size of random stochastic batch sampling, default=1000
        :attribute trainImages: vectorized zscores of training images
        :attribute trainLabels: vectorized labels of training images
        :attribute testImages: vectorized zscores of testing images
        :attribute testLabels: vectorized labels of testing images
        :attribute weights: vector matrix of pixel weights by classifications 785x10
        :attribute labels: 10x10 identity matrix used for label processing
        :attribute learn: learning rate, learn / (subset * 10)
        :attribute error: vector of testing errors on each label
        :attribute totals: vector of the number of each label in the testing set
        """
        self.trainImages, self.trainLabels = setupData('training')
        self.testImages, self.testLabels = setupData('testing')
        self.weights = np.array([[0.0] * len(self.trainImages[0]) for x in range(10)])
        self.labels = np.array([[1 if x == y else 0 for x in range(10)] for y in range(10)])
        self.learn = learn / (subset * 10)
        self.iterations = iterations
        self.subset = subset
        self.error = np.array([0] * 10)
        self.totals = np.array([0] * 10)
        for k in range(10):
            for _ in range(self.iterations):
                self.gradient(k)
        self.predict()

    def probability(self, x, k):
        """
        Determines probability of a label for an image
        :param x: image vector
        :param k: label
        :return: probability
        """
        dot = np.dot(self.weights[k], x)
        return 1 / (1 + math.exp(-1 * dot))

    def gradient(self, k):
        """
        Gradient descent learning algorithm, updates weights
        :param k: label
        """
        subset = random.sample(range(len(self.trainImages)), self.subset)
        temp = sum((self.probability(self.trainImages[i], [k]) - self.labels[k][self.trainLabels[i]])
                   * self.trainImages[i] for i in subset)
        self.weights[k] = self.weights[k] - self.learn * temp

    def predict(self):
        """
        Predicts testImages labels
        """
        for x, y in zip(self.testImages, self.testLabels):
            guess = 0
            best = 0
            self.totals[y] += 1
            for k in range(10):
                if best < self.probability(x, k):
                    guess = k
                    best = self.probability(x, k)
            if y != guess:
                self.error[y] += 1

class softmaxRegression:

    def __init__(self, learn=1, iterations=100, subset=1000):
        """
        Creates a softmax regression object and performs a gradient descent
        :param self: softmaxRegression object
        :param learn: learning rate factor, defaults=1
        :param iterations: number of weight training iterations, default=100
        :param subset: size of random stochastic batch sampling, default=1000
        :attribute trainImages: vectorized zscores of training images
        :attribute trainLabels: vectorized labels of training images
        :attribute testImages: vectorized zscores of testing images
        :attribute testLabels: vectorized labels of testing images
        :attribute weights: vector matrix of pixel weights by classifications 785x10
        :attribute labels: 10x10 identity matrix used for label processing
        :attribute learn: learning rate, learn / (subset * 10)
        :attribute error: vector of testing errors on each label
        :attribute totals: vector of the number of each label in the testing set
        """
        self.trainImages, self.trainLabels = setupData('training')
        self.testImages, self.testLabels = setupData('testing')
        self.weights = np.array([[0.0] * len(self.trainImages[0]) for x in range(10)])
        self.labels = np.array([[1 if x == y else 0 for x in range(10)] for y in range(10)])
        self.learn = learn / (subset * 10)
        self.iterations = iterations
        self.subset = subset
        self.error = np.array([0] * 10)
        self.totals = np.array([0] * 10)
        for _ in range(self.iterations):
            self.gradient()
        self.predict()

    def probability(self, x):
        """
        Determines probability of a label for an image
        :param x: image vector
        :return: probability vector 10x1
        """
        numerator = np.array([(math.exp(np.dot(self.weights[i], x))) for i in range(10)])
        denominator = sum(math.exp(np.dot(self.weights[i], x)) for i in range(10))
        return numerator / denominator

    def gradient(self):
        """
        Gradient descent learning algorithm, updates weights
        """
        subset = random.sample(range(len(self.trainImages)), self.subset)
        for k in range(10):
            temp = sum(self.trainImages[i] * (self.labels[k][self.trainLabels[i]]
                       - self.probability(self.trainImages[i])[k]) for i in subset)
            self.weights[k] = self.weights[k] - self.learn * (-1 * temp)

    def predict(self):
        """
        Predicts testImages labels
        """
        for x, y in zip(self.testImages, self.testLabels):
            self.totals[y] += 1
            if y != self.probability(x).argmax():
                self.error[y] += 1

def plotData():
    """
    Plots softmax accuracy relative to iterations of gradient descent
    """
    x = []
    y = []
    for iterations in range(10, 101, 10):
        sr = softmaxRegression(iterations=iterations)
        x.append(iterations)
        y.append(1 - sum(sr.error) / 2000)
    figure = plt.figure()
    ax = figure.add_subplot(1,1,1)
    ax.scatter(x, y)
    ax.set_title('Iterations vs. Test Accuracy')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Test Accuracy')
    figure.savefig('SoftmaxPlotAccuracy.png')




#builds logisticRegression object with defualt params
lr = logisticRegression()
#prints number of errors for each label type
print(lr.error)
#prints overall test accuracy
print('Logistic Regression Overall Accuracy: ', end='')
print(1 - sum(lr.error) / 2000)
count = 0
for x, y in zip(lr.error, lr.totals):
    print('Label ' + str(count) + ' Accuracy: ' + str(1 - x / y))
    count += 1
print(' ')

#Plots softmax accuracy data
plotData()

#build softmaxRegression object with default params
sr = softmaxRegression()
#prints number of errors for each label type
print(sr.error)
#prints overall test accuracy
print('Softmax Regression Overall Accuracy: ', end='')
print(1 - sum(sr.error) / 2000)
count = 0
for x, y in zip(sr.error, sr.totals):
    print('Label ' + str(count) + ' Accuracy: ' + str(1 - x / y))
    count += 1