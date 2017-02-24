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
    Train on entire multi-class MNIST dataset
    Weight initialization using Xavier Initialization
    tf.saver to save model at 25%, 50%, 75% and 100% of training process
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
        # Start to early stop at 5 for valid and 8 for test 
        self.numEpochNoDropout = 6
        # Early Stops at about 12 for both valid at test
        self.numEpochDropout = 12
        self.numEpoch = self.numEpochDropout
        # TEMP
        self.numEpoch = 1
        self.miniBatchSize = 500
        self.weightDecay = (3.0 * np.exp(1)) - 4.0
        self.dropoutProbability = 0.5
        # Size of array is number of layers
        # values are number of hidden units in each layer
        self.hiddenLayers = hiddenLayers
        # Add the final output as a hidden layer for multi-class (10 classes)
        self.hiddenLayers = np.append(hiddenLayers, 10)
        self.NeuralNetworkMethod()

    def ShuffleBatches(self, trainData, trainTarget):
        # Gets the state as the current time
        rngState = np.random.get_state()
        np.random.shuffle(trainData)
        np.random.set_state(rngState)
        np.random.shuffle(trainTarget)
        return trainData, trainTarget

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

        tf.add_to_collection("Weights", weight)

        # Add to collection for weight decay loss
        weightDecayCoeff = tf.div(tf.cast(tf.constant(self.weightDecay), "float32"),tf.constant(2.0))
        # Weight Decay Error calculation
        weightDecayMeanSquareError = tf.reduce_mean(tf.square(weight))
        weightDecayError = tf.multiply(weightDecayCoeff, weightDecayMeanSquareError)

        tf.add_to_collection("WeightDecayLoss", weightDecayError)

        return weightedSum

    def buildFullyConnectedNeuralNetwork(self):
        weightedSum  = tf.pack(self.trainData)

        X = tf.placeholder(tf.float32, [None, 784], name='X') 
        y_target = tf.placeholder(tf.float32, [None, 10], name='target_y')
        inputTensor = X

        for currLayer in self.hiddenLayers:
            weightedSum = self.layerWiseBuildingBlock(inputTensor, currLayer)
            # Parse with activation function of ReLu
            inputTensor = tf.nn.relu(weightedSum)
            # 2.4.1 Dropout
            inputTensor = tf.nn.dropout(inputTensor, self.dropoutProbability)

        # inputTensor is now the final hidden layer, but only need weighted Sum
        # Need add one more with softmax for output
        y_predicted = weightedSum 

        # Multi-class Classification
        correctPred = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_target, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, "float"))


        # Cross Entropy Softmax Error Multi-class Classification
        # note: Cross entropy only works with values from 0 to 1, so multi-class must be one hot encoded
        crossEntropySoftmaxError = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_predicted, y_target))
        adamOptimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate)

        # Calculate weight decay error
        totalWeightDecayError = sum(tf.get_collection("WeightDecayLoss"))
        crossEntropySoftmaxError = tf.add(crossEntropySoftmaxError, totalWeightDecayError)
        finalTrainingError = crossEntropySoftmaxError
        train = adamOptimizer.minimize(loss=finalTrainingError)

        weights = tf.get_collection("Weights")

        return X, y_target, y_predicted, finalTrainingError, train, accuracy, weights

    def NeuralNetworkMethod(self):
        maxTestClassificationAccuracy = 0.0
        inputTensor = tf.pack(self.trainData)
        # Build the fully connected Neural Network
        X, y_target, y_predicted, crossEntropyError, train, accuracy, weights = self.buildFullyConnectedNeuralNetwork()
        figureCount = 1 

        # Session
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)
        currEpoch = 0
        xAxis = []
        yTrainErr = []
        yValidErr = []
        yTestErr = []
        yTrainAcc = []
        yValidAcc = []
        yTestAcc = []
        numUpdate = 0
        step = 0
        errTrain = -1 
        errValid = -1 
        errTest = -1 
        accTrain = -1
        accValid = -1
        accTest = -1
        # TODO: 2.4.2 Save model from 25%, 50%, 75% and 100% from early stopping point
        while currEpoch <= self.numEpoch:
            self.trainData, self.trainTarget = self.ShuffleBatches(self.trainData, self.trainTarget)
            step = 0 
            while step*self.miniBatchSize < self.trainData.shape[0]: 
                # train comes from BuildGraph's optimization method
                # returnedValues = sess.run([whatYouWantToReturnThatWereReturnedFromBuildGraph], 
                #               feed_dic{valuesToFeedIntoPlaceHoldersThatWereReturnedFromBuildGraph})
                # sess.run() executes whatever graph you built once up to the point where it needs to fetch
                # and fetches everything that's in ([variablesToFetch])
                # Thus, if you don't fetch 'train = optimizer.minimize(loss)', it won't optimize it
                _, errTrain, yhat, accTrain, hiddenImages = sess.run([train, crossEntropyError, y_predicted, accuracy, weights], feed_dict={X: np.reshape(self.trainData[step*self.miniBatchSize:(step+1)*self.miniBatchSize], (self.miniBatchSize,784)),y_target: self.trainTarget[step*self.miniBatchSize:(step+1)*self.miniBatchSize]})
                step = step + 1
                numUpdate += 1
                # These will not optimize the function cause you did not fetch 'train' 
                # So it won't have to execute that.
                errValid, accValid = sess.run([crossEntropyError, accuracy], feed_dict={X: np.reshape(self.validData, (self.validData.shape[0],784)), y_target: self.validTarget})

                errTest, accTest = sess.run([crossEntropyError, accuracy], feed_dict={X: np.reshape(self.testData, (self.testData.shape[0], 784)), y_target: self.testTarget})
            # Plot against currEpoch for Neural Network
            xAxis.append(currEpoch)
            yTrainErr.append(errTrain)
            yTrainAcc.append(accTrain)
            yValidErr.append(errValid)
            yTestErr.append(errTest)
            yValidAcc.append(accValid)
            yTestAcc.append(accTest)
            currEpoch += 1
        print "LearningRate: " , self.learningRate, " Mini batch Size: ", self.miniBatchSize
        print "Iter: ", numUpdate
        print "Final Train MSE: ", errTrain
        print "Final Train Acc: ", accTrain
        print "Final Valid MSE: ", errValid
        print "Final Test MSE: ", errTest
        print "Final Valid Acc: ", accValid
        print "Final Test Acc: ", accTest
        import matplotlib.pyplot as plt
        plt.figure(figureCount)
        # Average across the 1000 hidden units to get a grascale image
        # To convert from (784,1000) to (784) to (28,28)
        hiddenImageToPlot = np.reshape(np.average(hiddenImages[0], 1), (28,28))
        plt.imshow(hiddenImageToPlot, interpolation="nearest", cmap="gray")
        plt.savefig("FirstHiddenLayerAverageEpoch" + str(self.numEpoch) + ".png")
        figureCount = figureCount + 1

        print 'Done Plotting AverageImage'
        plt.figure(figureCount)
        numHiddenLayer = hiddenImages[0].shape[1]
        fig = plt.figure(figsize=(28,28))
        numCol = 25
        numRow = numHiddenLayer/numCol
        for eachHiddenLayer in xrange(numHiddenLayer):
            print 'hl:', eachHiddenLayer
            # Draw each figures (28, 28)
            currImage = hiddenImages[0]
            hiddenImageToPlot = np.reshape(currImage[::,eachHiddenLayer:eachHiddenLayer+1], (28,28))
            ax = fig.add_subplot(numCol, numRow, eachHiddenLayer+1)
            ax.imshow(hiddenImageToPlot, interpolation="nearest", cmap="gray")
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.set_aspect(abs(x1-x0)/abs(y1-y0))
        print 'Saving all images'
        plt.savefig("FirstHiddenLayerAllUnitsEpoch" + str(self.numEpoch) + ".png")
        print 'Saved all images'

        figureCount = figureCount + 1
        plt.figure(figureCount)
        plt.plot(np.array(xAxis), np.array(yTrainErr))
        plt.savefig("TrainLossLearnRate" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')

        plt.figure(figureCount)
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yValidErr))
        plt.savefig("ValidLossLearnRate" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')
        plt.figure(figureCount)
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yTestErr))
        plt.savefig("TestLossLearnRate" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')

        plt.figure(figureCount)
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yTrainAcc))
        plt.savefig("TrainAccuracy" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')
        plt.figure(figureCount)
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yValidAcc))
        plt.savefig("ValidAccuracy" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')
        plt.figure(figureCount)
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yTestAcc))
        plt.savefig("TestAccuracy" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')
        return max(np.array(yTestAcc))

def convertOneHot(targetValues):
    numClasses = np.max(targetValues) + 1
    return np.eye(numClasses)[targetValues]

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
        trainTarget = convertOneHot(trainTarget)
        validTarget = convertOneHot(validTarget)
        testTarget = convertOneHot(testTarget)
        hiddenLayers = np.array([1000])
        for learningRate in [0.01]:
            tf.reset_default_graph()
            FullyConnectedNeuralNetwork(trainData, trainTarget, validData, validTarget, testData, testTarget, learningRate, hiddenLayers)
        sys.exit(0)
