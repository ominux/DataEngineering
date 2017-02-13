import numpy as np
import tensorflow as tf
import sys
import math

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
    def __init__(self, trainData, trainTarget, validData, validTarget, testData, testTarget, learningRate = 0.001, hiddenLayers = np.array([1000])):
        """
        hiddenLayers is an array indicating the number of hidden units in each layer
        """
        self.trainData = np.reshape(trainData, (trainData.shape[0], 784))
        self.trainTarget = trainTarget
        self.validData = validData
        self.validTarget = validTarget
        self.testData = testData
        self.testTarget = testTarget

        self.learningRate = learningRate
        # Size of array is number of layers,
        # values are number of hidden units in each layer
        self.hiddenLayers = hiddenLayers

        # Build the fully connected Neural Network
        self.buildFullyConnectedNeuralNetwork()

    def layerWiseBuildingBlock(self, inputTensor, numberOfHiddenUnits):
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
        numInput = inputTensor.get_shape().as_list()[1]
        numOutput = numberOfHiddenUnits
        # Xavier Initialization
        #variance = tf.div(tf.constant(3.0), tf.add(numInput, tf.constant(numOutput)))
        variance = 3.0/(numInput + numOutput)
        weight = tf.Variable(tf.truncated_normal(shape=[numInput, numOutput], stddev = math.sqrt(variance)))
        bias = tf.Variable(tf.zeros([numOutput]))
        weightedSum = tf.matmul(tf.cast(inputTensor, "float32"), weight) + bias
        return weightedSum

    def buildFullyConnectedNeuralNetwork(self):
        inputTensor = tf.pack(self.trainData)
        # TODO: may need to remove this for loop
        for currLayer in self.hiddenLayers:
            weightedSum = self.layerWiseBuildingBlock(inputTensor, currLayer)
            # Parse with activation function of ReLu
            inputTensor = tf.nn.relu(weightedSum)

if __name__ == "__main__":
    with np.load("notMNIST.npz") as data:
        Data, Target = data ["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        # Target values are from 0 to 9
        testData, testTarget = Data[16000:], Target[16000:]
        for learningRate in [0.01]:
            tf.reset_default_graph()
            FullyConnectedNeuralNetwork(trainData, trainTarget, validData, validTarget, testData, testTarget, learningRate, np.array([1000, 500]))
        sys.exit(0)
