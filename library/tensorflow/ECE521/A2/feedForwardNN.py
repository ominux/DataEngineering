import numpy as np
import tensorflow as tf
import sys

class FullyConnectedNeuralNetworks(object):
    """
    Fully Connected Neural Network
    Activation Function: Rectified Linear Unit
    Cost Function: Cross-Entropy
    Output Layer: Softmax 
    tf.saver to save model at 25%, 50%, 75% and 100% of training process
    Train on entire multi-class MNIST dataset
    Weight initialization using Xavier Initialization
    """
    def __init__(self, learningRate = 0.001, numHiddenLayers = 1, numHiddenUnits = 1000):
        self.learningRate = learningRate
        # TODO: Make this into an array where size of array is number of layers,
        #       and values are number of hidden units in each layer
        self.numHiddenLayers = numHiddenLayers
        self.numHiddenUnits = numHiddenUnits

    def weightedSum(self, inputTensor, numberOfHiddenUnits):
        """
        Input: Hidden activiations from previous layer
        Output: Weighted sum of the inputs
        Initialize Weight Matrix and Biases in same function
        Not Loops
        """
        weightedSum = 0.0
        # TODO: 
        return weightedSum

