import keras # Use keras for now
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import imdb
from keras.datasets import mnist
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models import word2vec

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

    def getText8(self):
        text8FilePath = '/downloadedDataset/text8'
        Text8Sentences = word2vec.Text8Corpus(text8FilePath)
        return Text8Sentences

    def getWikipedia(self):
        wikipediaFilePath = 'downloadedDataset/enwiki-latest-pages-articles.xml.bz2'
        wiki = WikiCorpus(wikipediaFilePath, lemmatize=False, dictionary={}) 
        wikipediaSentences = wiki.get_texts()
        return wikipediaSentences
