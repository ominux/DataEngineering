"""
This class implements kNearestNeighbour.
Assumes testData have shapes
(numTest, dataSize)
which must be consistent with
trainData with shape (numTrain, dataSize)
""" 
import numpy as np

class KNearestNeighbour(object):
    def __init__(self, trainData, trainLabel):
        self.trainData = trainData
        self.trainLabel = trainLabel

    def computeDistanceMatrix(self, testData):
        """
        Returns the distance matrix between every train and test instance
        Space Complexity = O(numTrain * numTest + numTrain * dataSize + numTest*dataSize)
        """
        numTrain = self.trainData.shape[0]
        numTest = testData.shape[0]
        trainDataSumSqr = np.sum(np.square(self.trainData), axis=1)
        testDataSumSqr = np.sum(np.square(testData), axis=1)
        trainMulTest= -2.0 * np.dot(testData, np.transpose(self.trainData))
        distanceMatrix = np.sqrt(
                np.reshape(testDataSumSqr, (-1, 1))
                + np.reshape(trainDataSumSqr, (1, -1))
                + trainMulTest)
        return distanceMatrix
    
    def predict(self, testData, k=1):
        """
        Gets the prediction of the given testData
        k is the number of instances to look at
        """
        distanceMatrix = self.computeDistanceMatrix(testData)
        closestIndices = np.argsort(distanceMatrix, axis=1)
        closestKIndices = closestIndices[:, :k]
        closestKLabels = self.trainLabel[closestKIndices]
        # NearestNeighbour only makes sense if using mode since labels may not be ordered
        from scipy.stats import mode
        majorityLabel, majorityCount = mode(closestKLabels, axis=1)
        majorityLabel = np.reshape(majorityLabel, (majorityLabel.size))
        return majorityLabel

    def accuracy(self, testData, testLabel, k):
        """
        Calculates the accuracy of the given testData. 
        """
        prediction = self.predict(testData, k)
        accuracy = np.sum(prediction == testLabel)/testLabel.size
        return accuracy
