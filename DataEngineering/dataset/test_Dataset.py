import unittest
import inspect

from DataEngineering.dataset import Dataset
from DataEngineering.algorithm import ConvolutionalNeuralNetwork
from DataEngineering.algorithm import LogisticRegression
from DataEngineering.algorithm import KNearestNeighbour

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("TestImports")
        self.dataset = Dataset()

    def testImportMnist(self):
        (xTrain, yTrain), (xTest, yTest) = self.dataset.getMnist()

    def testImportCifar10(self):
        (xTrain, yTrain), (xTest, yTest) = self.dataset.getCifar10()

    def testImportCifar100(self):
        (xTrain, yTrain), (xTest, yTest) = self.dataset.getCifar100()

    def testImportWikipedia(self):
        wikipediaSentences = self.dataset.getWikipedia()

    def testImportText8(self):
        text8Sentences = self.dataset.getText8()

    @classmethod
    def tearDownClass(self):
        print("Finished testing imports")

#(xTrain, yTrain), (xTest, yTest) = self.dataset.getImdb()
if __name__ == "__main__":
    unittest.main()
