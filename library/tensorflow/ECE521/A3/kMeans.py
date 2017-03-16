import tensorflow as tf
import numpy as np
from dataInitializer import DataInitializer

class KMeans(object):
    def __init__(self, K, trainData, validData, hasValid):
        """
        Constructor
        """
        self.K = K
        self.trainData = trainData
        self.validData = validData
        self.hasValid = hasValid

def executePartOne(questionTitle, K, dataType, hasValid):
    """
    Re-loads the data and re-randomize it with same seed anytime to ensure replicable results
    """
    print questionTitle
    trainData = 0
    validData = 0
    # Load data with seeded randomization
    dataInitializer = DataInitializer()
    if hasValid:
        trainData, validData = dataInitializer.getData(dataType, hasValid)
    else: 
        trainData = dataInitializer.getData(dataType, hasValid)
    # Execute algorithm 
    print "K:", K
    kObject = KMeans(K, trainData, validData, hasValid)

def executePartTwo(questionTitle, K, dataType, hasValid):
    print questionTitle
    trainData = 0
    validData = 0
    # Load data with seeded randomization
    dataInitializer = DataInitializer()
    trainData = dataInitializer.getData(dataType, hasValid)
    if hasValid:
        trainData, validData = dataInitializer.getData(dataType, hasValid)
    else: 
        trainData = dataInitializer.getData(dataType, hasValid)
    # Execute algorithm 
    print "K:", K
    kObject = KMeans(K, trainData, validData, hasValid)


if __name__ == "__main__":
    print "ECE521 Assignment 3: Unsupervised Learning: K Means"
    # Unsupervised => Data has no label or target
    questionTitle = "1.1.2"
    dataType = "2D"
    hasValid = False # No validation data
    K = 3
    executePartOne(questionTitle, K, dataType, hasValid)
    # '''

    '''
    questionTitle = "1.1.3"
    diffK = [1 2 3 4 5]
    dataType = "2D"
    hasValid = False
    for K in diffK:
        executePartOne(questionTitle, K, dataType, hasValid)
    # '''

    '''
    questionTitle = "1.1.4"
    diffK = [1 2 3 4 5]
    dataType = "2D"
    hasValid = True
    for K in diffK:
        executePartOne(questionTitle, K, dataType, hasValid)
    # '''

