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
        print("TrainDataShape", trainData.shape)
        print("TrainLabel", trainLabel.shape)
        knn = KNearestNeighbour(trainData, trainLabel)

    @classmethod
    def tearDownClass(self):
        pass

if __name__ == "__main__":
    unittest.main()
