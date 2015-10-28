import random


def perceptronLearning(teacher):
    """
    Perceptron Learning Rule
    :param teacher: dictionary of learning rules
    :return: possible solution for w1, w2, threshold
    """
    w1 = 0
    w2 = 0
    threshold = 0
    limit = 0
    while(limit < 100):
        x1 = random.randint(0,1)
        x2 = random.randint(0,1)
        net = x1 * w1 + x2 * w2
        if net >= threshold:
            output = 1
            if output != teacher[str(x1) + str(x2)]:
                #lower weights of active inputs, raise threshold
                if x1 == 1:
                    w1 -= 1
                if x2 == 1:
                    w2 -= 1
                threshold += 1
                limit = 0
            else:
                limit += 1
        else:
            output = 0
            if output != teacher[str(x1) + str(x2)]:
                #raise weights of active inputs, lower threshold
                if x1 == 1:
                    w1 += 1
                if x2 == 1:
                    w2 += 1
                threshold -= 1
                limit = 0
            else:
                limit += 1
    return [str(w1), str(w2), str(threshold)]

def findAllSolutions(teacher):
    """
    Determines all possible perceptron learning solutions
    :param teacher: dictionary of learning rules
    :return: array of possible solutions for perceptron learning rule
    """
    solutions = []
    limit = 0
    while(limit < 100):
        temp = perceptronLearning(teacher)
        if temp not in solutions:
            solutions += [temp]
        else:
            limit += 1
    return solutions

def showSolutions(solutions):
    """
    Used to print results to terminal
    :param solutions: array of possible solutions for perceptron learning rule
    """
    print('w1\tw2\tTheta')
    for x in solutions:
        print(x[0] + '\t' + x[1] + '\t' + x[2])
    return

#NAND Teacher
teacher = {'00': 1,
           '01': 1,
           '10': 1,
           '11': 0}

showSolutions(findAllSolutions(teacher))