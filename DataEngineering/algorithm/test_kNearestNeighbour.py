"""
This class tests the kNearestNeighbour
"""
import unittest
import numpy as np

from DataEngineering.algorithm import KNearestNeighbour

class TestKNearestNeighbour(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.smallerLabel = 5
        self.biggerLabel = 7
        self.trainData = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.trainLabel = np.array([self.smallerLabel, self.biggerLabel])
        self.knn = KNearestNeighbour(self.trainData, self.trainLabel)

    def testDistanceMatrix(self):
        testData = self.trainData
        distanceMatrix = self.knn.computeDistanceMatrix(testData)
        # Ensure that the diagonal elements are all 0
        assert np.sum(distanceMatrix.diagonal()) == 0.0

    def testPrediction(self):
        testData = self.trainData
        k = 1
        predictedLabels = self.knn.predict(testData, k)
        # Final predicted values should be same as train label
        # since using k = 1 with trainData to test
        assert np.array_equal(predictedLabels, self.trainLabel)

        # Check that it picks class with smallest value in case of tie
        k = 2
        predictedLabels = self.knn.predict(testData, k)
        assert np.array_equal(predictedLabels, np.array([self.smallerLabel, self.smallerLabel]))

    def testAccuracy(self):
        testData = self.trainData
        testLabel = self.trainLabel
        k = 1
        accuracy = self.knn.accuracy(testData, testLabel, k)
        assert accuracy == 1.0

    @classmethod
    def tearDownClass(self):
        pass

if __name__ == "__main__":
    unittest.main()
