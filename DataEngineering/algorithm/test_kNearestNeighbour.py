"""
This class tests the kNearestNeighbour
"""
import unittest
import numpy as np

from DataEngineering.algorithm import KNearestNeighbour

class TestKNearestNeighbour(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("TestKNearestNeighbour")

    def testDistanceMatrix(self):
        trainData = np.array([[1.0, 2.0], [3.0, 4.0]])
        trainLabel = np.array([0, 1])
        print(trainData.shape)
        print(trainLabel.shape)
        knn = KNearestNeighbour(trainData, trainLabel)

    @classmethod
    def tearDownClass(self):
        print("Finished testing KNearestNeighbour")

if __name__ == "__main__":
    unittest.main()
