"""
This class tests the kNearestNeighbour
"""
import unittest
import numpy as np

from DataEngineering.algorithm import KNearestNeighbour

class TestKNearestNeighbour(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    def testDistanceMatrix(self):
        trainData = np.array([[1.0, 2.0], [3.0, 4.0]])
        trainLabel = np.array([0, 1])
        knn = KNearestNeighbour(trainData, trainLabel)
        testData = trainData
        distanceMatrix = knn.computeDistanceMatrix(testData)
        # Ensure that the diagonal elements are all 0
        assert np.sum(distanceMatrix.diagonal()) == 0.0

    def testPrediction(self):
        trainData = np.array([[1.0, 2.0], [3.0, 4.0]])
        trainLabel = np.array([5, 7])
        testData = trainData
        k = 1
        knn = KNearestNeighbour(trainData, trainLabel)
        predictedLabels = knn.predict(testData, k)
        # Final predicted values should be same as train label
        # since using k = 1 with trainData to test
        assert np.array_equal(predictedLabels, trainLabel)

        # Check that it picks class with smallest value in case of tie
        k = 2
        predictedLabels = knn.predict(testData, k)
        assert np.array_equal(predictedLabels, np.array([5, 5]))

    @classmethod
    def tearDownClass(self):
        pass

if __name__ == "__main__":
    unittest.main()
