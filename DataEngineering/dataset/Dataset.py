import keras # Use keras for now
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import imdb
from keras.datasets import mnist

class Dataset(object):
    def __init__(self):
        self.file = "."

    def getCifar10(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return (x_train, y_train), (x_test, y_test)

    def getCifar100(self):
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        return (x_train, y_train), (x_test, y_test)

    def getImdb(self):
        (x_train, y_train), (x_test, y_test) = imdb.load_data()
        return (x_train, y_train), (x_test, y_test)

    def getMnist(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return (x_train, y_train), (x_test, y_test)


