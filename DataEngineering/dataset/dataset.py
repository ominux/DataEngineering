import os
import six
if six.PY3:
    from urllib.request import urlopen
else:
    from urllib2 import urlopen
import zipfile
import tarfile

from sklearn.datasets import fetch_lfw_people
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import imdb
from keras.datasets import mnist
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models import word2vec

class Dataset(object):
    def __init__(self):
        self.folder = "downloadedDataset"
        self.dataDir= os.path.expanduser(os.path.join('~', self.folder))
        if not os.path.exists(self.dataDir):
            os.makedirs(self.dataDir)

    def getCifar10(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return (x_train, y_train), (x_test, y_test)

    def getCifar100(self):
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        return (x_train, y_train), (x_test, y_test)

    def getMnist(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return (x_train, y_train), (x_test, y_test)

    def getLfw(self):
        lfwData = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        xTrain = lfwData['images']
        yTrain = lfwData['target']
        return (xTrain, yTrain)

    def getText8(self):
        text8URL = ("http://mattmahoney.net/dc/text8.zip")
        downloadedFilename = "text8.zip"
        text8FilePath = self.downloadFileIfDoesNotExist(text8URL, downloadedFilename)
        Text8Sentences = word2vec.Text8Corpus(text8FilePath)
        return Text8Sentences

    def getWikipedia(self):
        wikipediaURL = "http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
        downloadedFilename = "enwiki-latest-pages-articles.xml.bz2"
        wikipediaFilePath = self.downloadFileIfDoesNotExist(wikipediaURL, downloadedFilename)
        wiki = WikiCorpus(wikipediaFilePath, lemmatize=False, dictionary={}) 
        wikipediaSentences = wiki.get_texts()
        return wikipediaSentences

    def getImdb(self):
        (x_train, y_train), (x_test, y_test) = imdb.load_data()
        return (x_train, y_train), (x_test, y_test)


    def downloadFileIfDoesNotExist(self, downloadUrl, downloadedFileName):
        """
        Returns the final path to open the file for processing to get the information needed
        """
        extractType = None
        if "zip" in downloadUrl:
            extractType = 'zip'
        elif "tar" in downloadUrl and "gz" not in downloadUrl:
            extractType = 'tar'
        downloadedFilePath= os.path.join(self.dataDir, downloadedFileName)
        finalFilePath = downloadedFilePath
        if extractType is not None:
            # Remove the .zip or .tar extension
            if extractType is 'zip' or extractType is 'tar':
                finalFilePath = os.path.join(self.dataDir, downloadedFileName[:-4])
            elif extractType is 'tar.gz':
                finalFilePath = os.path.join(self.dataDir, downloadedFileName[:-7])
        # Only download if final file isn't there
        if not os.path.exists(finalFilePath):
            print("File " + finalFilePath + " doesn't exist, downloading...")
            # Only download if compressed file isn't already there
            if not os.path.exists(downloadedFilePath):
                print("File " + downloadedFilePath + " doesn't exist, downloading...")
                opener = urlopen(downloadUrl)
                with open(downloadedFilePath, 'wb') as f:
                    f.write(opener.read())
            # Extract files if needed
            if extractType is 'zip':
                with zipfile.ZipFile(downloadedFilePath) as archive:
                    archive.extractall(self.dataDir)
            elif extractType is 'tar':
                with tarfile.open(downloadedFilePath) as archive:
                    archive.extractall(self.dataDir)
            # Remove any temporary files
            if downloadedFilePath != finalFilePath:
                os.remove(downloadedFilePath)
            # TODO: Handle .tar.gz files
        return finalFilePath
