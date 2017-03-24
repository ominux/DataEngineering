import tensorflow as tf
import numpy as np
import sys
from dataInitializer import DataInitializer
from utils import * 
import datetime
import sys
import matplotlib.pyplot as plt

class FactorAnalysis(object):
    def __init__(self, questionTitle, K, trainData, trainTarget, validData, validTarget, testData, testTarget, numEpoch = 500, learningRate = 0.001): 
        """
        Constructor
        """
        self.K = K # number of factors
        self.trainData = trainData
        self.trainTarget = trainTarget
        self.validData = validData 
        self.validTarget = validTarget
        self.testData = testData
        self.testTarget = testTarget
        self.D = self.trainData[0].size # Dimension of each data
        self.learningRate = learningRate
        self.numEpoch = numEpoch
        self.miniBatchSize = self.trainData.shape[0] # miniBatchSize is entire data size
        self.questionTitle = questionTitle
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate, beta1=0.9, beta2=0.99, epsilon=1e-5)
        # Execute Factor Analysis
        self.FactorAnalysisMethod()

    def saveGrayscaleImage(self, image, width=8, height=8, imageName=""):
        """ This plots an image given its width and height 
        image is the image to plot
        imageName is the name of the image to save as.
        """
        figureCount = 0 # TODO: Make global
        plt.figure(figureCount)
        # Draw each figures (8, 8)
        currImage = image[:]
        currImage = np.reshape(currImage, (width,height))
        plt.imshow(currImage, interpolation="nearest", cmap="gray")
        plt.savefig(str(imageName) + ".png")

    def printTensor(self, tensorToPrint, trainData, message=""):
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)
        printDict = {trainData: self.trainData}
        valueToPrint = sess.run([tensorToPrint], feed_dict = printDict)
        print message, valueToPrint
        print "shape", np.array(valueToPrint).shape
        plt.close()
        plt.clf()

    def printPlotResults(self, xAxis, yTrainErr, yValidErr, numUpdate, currTrainDataShuffle, factorMean, factorCovariance, factorWeights):
        figureCount = 0 # TODO: Make global
        import matplotlib.pyplot as plt
        print "mean", factorMean
        print "K: ", self.K
        print "Iter: ", numUpdate
        print "mean", factorMean
        print "meanShape", factorMean.shape
        print "CoVariance", factorCovariance
        print "CoVarianceShape", factorCovariance.shape
        print "Lowest TrainLoss", np.min(yTrainErr)
        print "Lowest ValidLoss", np.min(yValidErr)

        trainStr = "Train"
        validStr = "Valid"
        typeLossStr = "Loss"
        typeScatterStr = "Assignments"
        trainLossStr = trainStr + typeLossStr
        validLossStr = validStr + typeLossStr
        iterationStr = "Iteration"
        paramStr = "K" + str(self.K) + "Learn" + str(self.learningRate) + "NumEpoch" + str(self.numEpoch)

        # Train Loss
        figureCount = figureCount + 1
        plt.figure(figureCount)
        title = trainStr + typeLossStr + paramStr
        plt.title(title)
        plt.xlabel(iterationStr)
        plt.ylabel(typeLossStr)
        plt.plot(np.array(xAxis), np.array(yTrainErr), label = trainLossStr)
        plt.legend()
        plt.savefig(self.questionTitle + title + ".png")
        plt.close()
        plt.clf()

        # Valid Loss
        figureCount = figureCount + 1
        plt.figure(figureCount)
        title = validStr + typeLossStr + paramStr
        plt.title(title)
        plt.xlabel(iterationStr)
        plt.ylabel(typeLossStr)
        plt.plot(np.array(xAxis), np.array(yValidErr), label = validLossStr)
        plt.legend()
        plt.savefig(self.questionTitle + title + ".png")
        plt.close()
        plt.clf()

        # Weight Images
        for i in xrange(self.K):
            imageTitle = self.questionTitle + "WeightDim" + str(i) + "K" + str(self.K) +  "NumEpoch" + str(self.numEpoch)
            self.saveGrayscaleImage(factorWeights[:, i], 8, 8, imageTitle)

    def FactorAnalysisMethod(self):
        ''' 
        Build Graph and execute in here
        so don't have to pass variables one by one
        Bad Coding Style but higher programmer productivity
        '''
        trainData = tf.placeholder(tf.float32, shape=[None, self.D], name="trainingData")
        validData = tf.placeholder(tf.float32, shape=[None, self.D], name="validationData")
        # Build Graph 
        print "trainShape", self.trainData.shape
        print "validShape", self.validData.shape
        print "testShape", self.testData.shape
        factorMean = tf.Variable(tf.truncated_normal([self.D]))
        factorWeights = tf.Variable(tf.truncated_normal([self.D, self.K]))
        factorStdDeviationConstraint = tf.Variable(tf.truncated_normal([self.D]))
        factorTraceCoVariance = tf.exp(factorStdDeviationConstraint)
        factorCovariance = tf.diag(factorTraceCoVariance) + tf.matmul(factorWeights, tf.transpose(factorWeights))
        factorCovarianceInv = tf.matrix_inverse(factorCovariance)
        logDeterminantCovariance = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(factorCovariance))))

        # Train Loss
        xDeductU = tf.subtract(trainData, factorMean)
        xDeductUTranspose = tf.transpose(xDeductU)
        total = tf.trace(tf.matmul(tf.matmul(xDeductU, factorCovarianceInv), xDeductUTranspose))
        logProbability = -0.5 * (total + self.D * tf.log(2.0 * np.pi) + logDeterminantCovariance)
        loss = tf.negative(logProbability)

        validDeductU = tf.subtract(trainData, factorMean)
        validDeductUTranspose = tf.transpose(validDeductU)
        validTotal = tf.trace(tf.matmul(tf.matmul(validDeductU, factorCovarianceInv), validDeductUTranspose))
        validLogProbability = -0.5 * (validTotal + self.D * tf.log(2.0 * np.pi) + logDeterminantCovariance)
        validLoss = tf.negative(validLogProbability)

        train = self.optimizer.minimize(loss)
        # Session
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)
        currEpoch = 0
        minAssignTrain = 0
        minAssignValid = 0
        centers = 0
        xAxis = []
        yTrainErr = []
        yValidErr = []
        numUpdate = 0
        step = 0
        currTrainDataShuffle = self.trainData
        while currEpoch < self.numEpoch:
            np.random.shuffle(self.trainData) # Shuffle Batches
            step = 0
            while step*self.miniBatchSize < self.trainData.shape[0]:
                feedDicts = {trainData: self.trainData[step*self.miniBatchSize:(step+1)*self.miniBatchSize], validData: self.validData}
                _, errTrain, errValid, paramFactorMean, paramFactorCovariance, paramFactorWeights = sess.run([train, loss, validLoss, factorMean, factorCovariance, factorWeights], feed_dict = feedDicts)
                xAxis.append(numUpdate)
                yTrainErr.append(errTrain)
                yValidErr.append(errValid)
                step += 1
                numUpdate += 1
            currEpoch += 1

            if currEpoch%100 == 0:
                logStdOut("e: " + str(currEpoch))
        # Calculate everything again without training
        feedDicts = {trainData: self.trainData, validData: self.validData}
        errTrain, errValid, paramFactorMean, paramFactorCovariance, paramFactorWeights = sess.run([loss, validLoss, factorMean, factorCovariance, factorWeights], feed_dict = feedDicts)
        # Count how many assigned to each class
        currTrainDataShuffle = self.trainData
        self.printPlotResults(xAxis, yTrainErr, yValidErr, numUpdate, currTrainDataShuffle, paramFactorMean, paramFactorCovariance, paramFactorWeights)

def executeFactorAnalysis(questionTitle, K, numEpoch, learningRate):
    """
    Re-loads the data and re-randomize it with same seed anytime to ensure replicable results
    """
    logStdOut(questionTitle)
    print questionTitle
    trainData = 0
    validData = 0
    # Load data with seeded randomization
    dataInitializer = DataInitializer()
    trainData, trainTarget, validData, validTarget, testData, testTarget = dataInitializer.getTinyData()

    # Execute algorithm 
    kObject = FactorAnalysis(questionTitle, K, trainData, trainTarget, validData, validTarget, testData, testTarget, numEpoch, learningRate)
    logElapsedTime(questionTitle + "K" + str(K) + "NumEpoch" + str(numEpoch))

# Global for logging
questionTitle = "" # Need to be global for logging to work
startTime = datetime.datetime.now()
figureCount = 1 # To not overwrite existing pictures

def logStdOut(message):
    # Temporary print to std out
    # sys.stdout = sys.__stdout__ # TODO: Uncomment this
    print message
    # Continue editing same file
    # sys.stdout = open("result" + questionTitle + ".txt", "a") #TODO: Uncomment this

def logElapsedTime(message):
    ''' Logs the elapsedTime with a given message '''
    global startTime 
    endTime = datetime.datetime.now()
    elapsedTime = endTime - startTime
    hours, remainder = divmod(elapsedTime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    totalDays = elapsedTime.days
    timeStr = str(message) + ': Days: ' + str(totalDays) +  " hours: " + str(hours) + ' minutes: ' + str(minutes) +  ' seconds: ' + str(seconds)
    logStdOut(timeStr)
    startTime = datetime.datetime.now()

if __name__ == "__main__":
    print "ECE521 Assignment 3: Unsupervised Learning: Factor Analysis"
    questionTitle = "3.1.2"
    numEpoch = 200
    learningRate = 0.1
    K = 4
    executeFactorAnalysis(questionTitle, K, numEpoch, learningRate)
    # '''
    
    '''
    questionTitle = "3.1.3"
    diffK = [1, 2, 3, 4, 5]
    # for K in diffK:
        executeFactorAnalysis(questionTitle, K, numEpoch, learningRate)
    # TODO:
    # '''

