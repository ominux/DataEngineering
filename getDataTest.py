'''
# >> python3 getDataTest.py
# This file shows how to open all the dataset using python3
import unittest
import numpy as np
import os

class TestCifar10(unittest.TestCase):
    """
    data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

    The dataset contains another file, called batches.meta. It too contains a Python dictionary object. It has the following entries:
        label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc
    """
    @classmethod
    def setUpClass(self):
        print("TestCifar10")
        self.folder = os.path.expanduser("~/.keras/datasets/cifar-10-batches-py")

    def testLoadableClassData(self):
        print("TestLoadableClassData")
        import pickle
        filee = "batches.meta"
        with open(self.folder + "/" + filee, 'rb') as fo:
            dictionary = pickle.load(fo)
            print(dictionary.keys())
        #  A bunch of png files
        for key in dictionary:
            print("Key:", key)
            print("Value:", dictionary[key])

    def testLoadableDictionaryData(self):
        print("TestLoadableDictionaryData")
        import pickle
        files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"] 
        for filee in files:
            with open(self.folder + "/" + filee, 'rb') as fo:
                dictionary = pickle.load(fo)
                print(dictionary.keys())
                for key in dictionary.keys():
                    print("Key: ", key)
                    print("Value: ", dictionary[key][0:2]) # Print only 2 instances of data
            break # First batch is enough

    def testGetTrainTestData(self):
        print("TestGetTrainTestData")
        import pickle
        files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"] 
        xAppend = []
        yAppend = []
        # TODO
        '''
        for filee in files:
            with open(self.folder + "/" + filee, 'rb') as fo:
                dictionary = pickle.load(fo)
                X = dictionary['data']
                Y =  dictionary['labels']
                X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
                Y = np.array(Y)
                xAppend.append(X)
                yAppend.append(Y)
        xTrain = np.concatenate(xAppend)
        yTrain = np.concatenate(yTrain)
        with open(self.folder + "/" + "test_batch", 'rb') as fo:
            dictionary = pickle.load(fo)
        print(dictionary.key())
        X = dictionary['data']
        Y = dictionary['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        xTest = X
        yTest = Y
        #xTrain, yTrain, xTest, yTest
        '''

    @classmethod
    def tearDownClass(self):
        print("Finished testing Cifar10")

class TestCifar100(unittest.TestCase):
    """
    This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).
    Here is the list of classes in the CIFAR-100:
        Superclass  Classes
        aquatic mammals beaver, dolphin, otter, seal, whale
        fish    aquarium fish, flatfish, ray, shark, trout
        flowers orchids, poppies, roses, sunflowers, tulips
        food containers bottles, bowls, cans, cups, plates
        fruit and vegetables    apples, mushrooms, oranges, pears, sweet peppers
        household electrical devices    clock, computer keyboard, lamp, telephone, television
        household furniture bed, chair, couch, table, wardrobe
        insects bee, beetle, butterfly, caterpillar, cockroach
        large carnivores    bear, leopard, lion, tiger, wolf
        large man-made outdoor things   bridge, castle, house, road, skyscraper
        large natural outdoor scenes    cloud, forest, mountain, plain, sea
        large omnivores and herbivores  camel, cattle, chimpanzee, elephant, kangaroo
        medium-sized mammals    fox, porcupine, possum, raccoon, skunk
        non-insect invertebrates    crab, lobster, snail, spider, worm
        people  baby, boy, girl, man, woman
        reptiles    crocodile, dinosaur, lizard, snake, turtle
        small mammals   hamster, mouse, rabbit, shrew, squirrel
        trees   maple, oak, palm, pine, willow
        vehicles 1  bicycle, bus, motorcycle, pickup truck, train
        vehicles 2  lawn-mower, rocket, streetcar, tank, tractor
    """
    @classmethod
    def setUpClass(self):
        print("TestCifar100")
        self.folder = os.path.expanduser("~/.keras/datasets/cifar-100-python")

    def testLoadableClassData(self):
        print("TestLoadableClassData")
        import pickle
        filee = "meta"
        with open(self.folder + "/" + filee, 'rb') as fo:
            dictionary = pickle.load(fo)
            print(dictionary.keys())
        #  A bunch of png files
        for key in dictionary:
            print("Key:", key)
            print("Value:", dictionary[key])

    def testLoadableDictionaryData(self):
        print("TestLoadableDictionaryData")
        # Load CIFAR100 data
        import pickle
        files = ["train", "test"]
        for filee in files:
            with open(self.folder + "/" + filee, 'rb') as fo:
                dictionary = pickle.load(fo)
                print(dictionary.keys())
                for key in dictionary.keys():
                    print("Key: ", key)
                    print("Value: ", dictionary[key][0:2]) # Print only 2 instances of data
            break # Train and test data are same format

    @classmethod
    def tearDownClass(self):
        print("Finished testing Cifar100")


if __name__ == "__main__":
    """
    Test that all dataset files can be obtained successfully. 
    """
    unittest.main()
'''
