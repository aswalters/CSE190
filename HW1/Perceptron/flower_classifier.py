import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def findZScore(data):
    """
    Predicts Z-Scores of data
    :param data: List of x vectors and labels ex: [[array[], label],...]
    :return: data with x vectors normalized to Z-Scores
    """
    for col in range(4):
        temp = []
        for row in range(len(data)):
            temp.append(data[row][0][col])
        zScores = stats.zscore(np.array(temp))
        for row in range(len(data)):
            data[row][0][col] = zScores[row]
    return data

def plotData(data, x, y, name, zScore):
    """
    Scatter plots two columns in data
    :param data: List of x vectors and labels ex: [[array[], label],...]
    :param x: column in data
    :param y: column in data
    :param name: list of attribute name for x, y
    :param zScore: flag if data has been normailized to Z-Scores
    """
    xAxis, yAxis = name
    figure = plt.figure()
    ax = figure.add_subplot(1,1,1)
    xSet, ySet, xVer, yVer = [[],[],[],[]]
    for line in data:
        if line[1] == 0:
            xSet.append(line[0][x])
            ySet.append(line[0][y])
        else:
            xVer.append(line[0][x])
            yVer.append(line[0][y])
    ax.scatter(xSet, ySet, color='red')
    ax.scatter(xVer, yVer, color='blue')
    if zScore:
        ax.set_title('Z Scores: ' + xAxis + ' vs. ' + yAxis +
                     '(Setosa=Red, Versicolor=Blue)')
    else:
        ax.set_title(xAxis + ' vs. ' + yAxis +
                     '(Setosa=Red, Versicolor=Blue)')
    ax.set_xlabel(xAxis)
    ax.set_ylabel(yAxis)
    if zScore:
        figure.savefig('ScatterPlots/ZScore/ZScore_' + xAxis + '_' +
                       yAxis + '.png')
    else:
        figure.savefig('ScatterPlots/' + xAxis + '_' + yAxis + '.png')
    return

def generatePlots(inputFile, zScore):
    """
    Builds all possible column vs. column plots
    :param inputFile: filename
    :param zScore: flag if data has been normailized to Z-Scores
    """
    name = {0: 'Sepal Length',
            1: 'Sepal Width',
            2: 'Pedal Length',
            3: 'Pedal Width'}
    data = parseData(inputFile)
    if zScore:
        data = findZScore(data)
    for i in range(4):
        for j in range(i+1, 4):
            if i != j:
                plotData(data, i, j, [name[i], name[j]], zScore)
    return

def classLabel(label):
    """
    Determines label of attributes
    :param label: flower name
    :return: binary label
    """
    if 'setosa' in label:
        return 0
    else:
        return 1

def parseData(trainFile):
    """
    Parses data out of input file
    :param trainFile: filename
    :return: List of x vectors and labels ex: [[array[], label],...]
    """
    file = open(trainFile)
    lines = file.readlines()
    for x in range(len(lines)):
        line = lines[x].strip().split(',')
        xVector = np.array([float(line[x]) for x in range(4)])
        lines[x] = [xVector, classLabel(line[4])]
    file.close()
    return lines

def trainPerceptron(trainFile, zScore):
    """
    Trains w vector and threshold
    :param trainFile: filename
    :param zScore: flag if you want the data to be normailized to Z-Scores
    :return: w and threshold
    """
    a = random.randint(1, 1000)
    train = parseData(trainFile)
    if zScore:
        train = findZScore(train)
    w = np.array([0] * 4)
    threshold = 0
    randRange = len(train) - 1
    limit = 0
    while (limit < 100):
        rand = random.randint(0,randRange)
        x, teacher = train[rand]
        net = np.dot(x, w)
        if net >= threshold:
            output = 1
        else:
            output = 0
        w = w + a * (teacher - output) * x
        threshold = threshold + (teacher - output)
        limit += 1
    return w, threshold

def predict(w, threshold, testFile, zScore):
    """
    Predicts type of flower and prints error rate
    :param w: vector of weights
    :param threshold: threshold constant
    :param testFile: filename
    :param zScore: flag if w and threshold were generated based on Z-Scores
    """
    test = parseData(testFile)
    if zScore:
        test = findZScore(test)
    error = 0
    for x, label in test:
        net = np.dot(x, w)
        if net >= threshold:
            output = 1
        else:
            output = 0
        if output != label:
            error += 1
    print('Error Rate: ', end='')
    print(error / len(test))
    return

#Builds scatter plots
generatePlots('iris_train.data', False)
#Builds Z-Score scatter plots
generatePlots('iris_train.data', True)
#Trains perceptron
w, threshold = trainPerceptron('iris_train.data', False)
#Predicts results with perceptron
predict(w, threshold, 'iris_test.data', False)