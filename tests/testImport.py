import inspect

from DataEngineering.dataset import Dataset
from DataEngineering.algorithm import ConvolutionalNeuralNetwork
from DataEngineering.algorithm import LogisticRegression

dataset = Dataset()
print("Importing... MNIST")
(xTrain, yTrain), (xTest, yTest) = dataset.getMnist()
print("Importing... Cifar10")
(xTrain, yTrain), (xTest, yTest) = dataset.getCifar10()
print("Importing... Cifar100")
(xTrain, yTrain), (xTest, yTest) = dataset.getCifar100()
print("Importing... imdb")
(xTrain, yTrain), (xTest, yTest) = dataset.getImdb()
print("Importing... wikipedia")
wikipediaSentences = dataset.getWikipedia()
print("Importing... text8")
text8Sentences = dataset.getText8()

lala = ConvolutionalNeuralNetwork()
