import inspect

from DataEngineering.dataset import Dataset
from DataEngineering.algorithm import ConvolutionalNeuralNetwork
from DataEngineering.algorithm import LogisticRegression

dataset = Dataset()
(xTrain, yTrain), (xTest, yTest) = dataset.getMnist()
(xTrain, yTrain), (xTest, yTest) = dataset.getCifar10()
(xTrain, yTrain), (xTest, yTest) = dataset.getCifar100()
(xTrain, yTrain), (xTest, yTest) = dataset.getImdb()

lala = ConvolutionalNeuralNetwork()

