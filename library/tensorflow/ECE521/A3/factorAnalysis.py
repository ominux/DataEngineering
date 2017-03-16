import tensorflow as tf
import numpy as np
from dataInitializer import DataInitializer

class FactorAnalysis(object):
    def __init__(self, K, trainData, validData, hasValid):
        """
        Constructor
        """
        self.K = K
        self.trainData = trainData
        self.validData = validData
        self.hasValid = hasValid

if __name__ == "__main__":
    print "ECE521 Assignment 3: Unsupervised Learning: Factor Analysis"

    '''
    questionTitle = "3.1.2"
    # TODO:
    questionTitle = "3.1.3"
    # TODO:
    '''
