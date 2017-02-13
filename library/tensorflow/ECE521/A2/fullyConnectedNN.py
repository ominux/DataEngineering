import numpy as np
import tensorflow as tf
import sys

class FullyConnectedNeuralNetwork(object):
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
        Layer-wise Building Block
        Input: 
            inputTensor: Hidden activations from previous layer
            numberOfHiddenUnitsInCurrentLayer
        Output: z = Weighted sum of the inputs

        Initialize Weight Matrix and Biases in this function
        A list of hidden activations
        No Loops
        """
        numInput = inputTensor.shape[0]
        numOutput = numberOfHiddenUnits
        # Xavier Initialization
        variance = 3.0/(numInput + numOutput)
        # TODO: Calculate squareroot of variance
        weight = tf.Variable(tf.truncated_normal(shape=[numInput, numOutput], stddev= variance))
        bias = tf.Variable(tf.zeros([numOutput]))
        weightedSum = tf.matmul(inputTensor, weight) + bias
        return weightedSum

if __name__ == "__main__":
    FullyConnectedNeuralNetwork(0.001, 1, 1000)
    sys.exit(0)
