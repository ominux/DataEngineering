import numpy as np
import tensorflow as tf

class DataInitializer(object):
    def __init__(self):
        """
        Data Initializer code
        """
        self.data2D = np.load("data2D.npy")
        self.data100D = np.load("data100D.npy")
        self.tinyData = np.load("tinymnist.npz")
    
    def getData(self, dataType, hasValid):
        """
        Returns the train, validation, and test data for 2D dataset 
        that has already been randomized
        """
        data = 0
        if dataType == "2D":
            data = self.data2D
        elif dataType == "100D":
            data = self.data100D
        if hasValid:
            trainData, validData = self.splitDataRandom(data, hasValid)
            '''
            print 'data' + str(dataType) + ' All: (number of data, data dimension):', data.shape
            print 'data' + str(dataType) + ' Train: (number of data, data dimension):', trainData.shape
            print 'data' + str(dataType) + ' Valid: (number of data, data dimension):', validData.shape
            '''
            return trainData, validData
        trainData = self.splitDataRandom(data, hasValid)
        '''
        print 'data' + str(dataType) + ' All: (number of data, data dimension):', data.shape
        print 'data' + str(dataType) + ' Train: (number of data, data dimension):', trainData.shape
        '''
        return trainData

    def getTinyData(self):
        """
        Returns the train, validation, and test data for tinyMnist dataset 
        that has already been randomized.
        """
        trainData, trainTarget = self.tinyData["x"], self.tinyData["y"]
        validData, validTarget = self.tinyData["x_valid"], self.tinyData ["y_valid"]
        testData, testTarget = self.tinyData["x_test"], self.tinyData["y_test"]
        print trainData.shape
        print trainTarget.shape
        print validData.shape
        print validTarget.shape
        print testData.shape
        print testTarget.shape
        # TODO: Randomize data? 
        return trainData, trainTarget, validData, validTarget, testData, testTarget

    def splitDataRandom(self, data, hasValid):
        """
        Splits data using the ratio:
        i) If it hasValid
            66.6% training data
            33.3% validation data
        ii) No validation data
            100.0% training data
        """
        np.random.seed(521)
        randIdx = np.arange(len(data))
        np.random.shuffle(randIdx)
        if hasValid:
            trainStopPoint = int(np.ceil((2.0/3.0)*len(data)))
            # print trainStopPoint
            trainData, validData = data[randIdx[:trainStopPoint]], data[randIdx[trainStopPoint:]]
            return trainData, validData
        else:
            trainData = data[randIdx[:]]
            return trainData
