import tensorflow as tf
import numpy as np
import sys
from dataInitializer import DataInitializer
from utils import * 
import datetime
import sys

class MixtureOfGaussians(object):
    def __init__(self, questionTitle, K, trainData, validData, hasValid, dataType, numEpoch = 500, learningRate = 0.001): 
        """
        Constructor
        """
        self.K = K # number of clusters
        self.dataType = dataType
        self.trainData = trainData
        self.validData = validData 
        self.D = self.trainData[0].size # Dimension of each data
        self.hasValid = hasValid
        self.learningRate = learningRate
        self.numEpoch = numEpoch
        self.miniBatchSize = self.trainData.shape[0] # miniBatchSize is entire data size
        self.questionTitle = questionTitle
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate, beta1=0.9, beta2=0.99, epsilon=1e-5)
        # Execute Mixture of Gaussians
        self.MixtureOfGaussiansMethod()

    def printPlotResults(self, xAxis, yTrainErr, yValidErr, numUpdate, minAssignTrain, currTrainData, clusterMean, clusterStdDeviation, clusterPrior,  minAssignValid):
        figureCount = 0 # TODO: Make global
        import matplotlib.pyplot as plt
        if self.dataType == "2D":
            print "mean", clusterMean
        print "K: ", self.K
        print "Iter: ", numUpdate
        numTrainAssignEachClass = np.bincount(minAssignTrain)
        numValidAssignEachClass = np.bincount(minAssignValid)
        print "Train Assignments To Classes:", numTrainAssignEachClass
        percentageTrainAssignEachClass = numTrainAssignEachClass/float(sum(numTrainAssignEachClass))
        print "Train Percentage Assignment To Classes:", percentageTrainAssignEachClass
        percentageValidAssignEachClass = percentageTrainAssignEachClass # Initialize
        if self.hasValid:
            print "Valid Assignments To Classes:", numValidAssignEachClass
            percentageValidAssignEachClass = numValidAssignEachClass/float(sum(numValidAssignEachClass))
            print "Valid Percentage Assignment To Classes:", percentageValidAssignEachClass
        print "prior", clusterPrior
        print "prior.shape", clusterPrior.shape
        print "prior Sum", np.sum(clusterPrior)
        print "stdDeviation", clusterStdDeviation
        print "stdDeviationShape", clusterStdDeviation.shape
        print str(self.K) + "Lowest TrainLoss", np.min(yTrainErr)
        print str(self.K) + "Lowest ValidLoss", np.min(yValidErr)

        trainStr = "Train"
        validStr = "Valid"
        typeLossStr = "Loss"
        typeScatterStr = "Assignments"
        trainLossStr = trainStr + typeLossStr
        validLossStr = validStr + typeLossStr
        iterationStr = "Iteration"
        dimensionOneStr = "D1"
        dimensionTwoStr = "D2"
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
        if self.hasValid:
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

        if self.dataType != "2D":
            return
        # Plot percentage in each different classes as well
        # Scatter plot based on assignment colors
        # Including percentage as the label
        # Train Scatter Plot
        figureCount = figureCount + 1
        plt.figure(figureCount)
        plt.axes()
        title = trainStr + typeScatterStr + paramStr
        plt.title(title)
        plt.xlabel(dimensionOneStr)
        plt.ylabel(dimensionTwoStr)
        colors = ['blue', 'red', 'green', 'black', 'yellow']
        plt.scatter(currTrainData[:, 0], currTrainData[:, 1], c=minAssignTrain, s=10, alpha=0.5)
        colors = colors[:self.K]
        for i, j, k, l in zip(clusterMean, percentageTrainAssignEachClass, colors, clusterStdDeviation[0]):
            plt.plot(i[0], i[1], 'kx', markersize=15, label=j, c=k)
            circle = plt.Circle((i[0], i[1]), radius=2*l, color=k, fill=False)
            plt.gca().add_patch(circle)
        plt.legend()
        plt.savefig(self.questionTitle + title + ".png")
        plt.close()
        plt.clf()

        if self.hasValid:
            # Valid Scatter Plot
            figureCount = figureCount + 1
            plt.figure(figureCount)
            plt.axes()
            title = validStr + typeScatterStr + paramStr
            plt.title(title)
            plt.xlabel(dimensionOneStr)
            plt.ylabel(dimensionTwoStr)
            colors = ['blue', 'red', 'green', 'black', 'yellow']
            plt.scatter(self.validData[:, 0], self.validData[:, 1], c=minAssignValid, s=10, alpha=0.5)
            colors = colors[:self.K]
            for i, j, k, l in zip(clusterMean, percentageValidAssignEachClass, colors, clusterStdDeviation[0]):
                plt.plot(i[0], i[1], 'kx', markersize=15, label=j, c=k)
                circle = plt.Circle((i[0], i[1]), radius=2*l, color=k, fill=False)
                plt.gca().add_patch(circle)
            plt.legend()
            plt.savefig(self.questionTitle + title + ".png")
            plt.close()
            plt.clf()

    def PairwiseDistances(self, X, U):
        """
        input:
            X is a matrix of size (B x D)
            U is a matrix of size (K x D)
        output:
            Distances = matrix of size (B x K) containing the pairwise Euclidean distances
        """
        batchSize = tf.shape(X)[0] 
        dimensionSize = tf.shape(X)[1]
        numClusters = tf.shape(U)[0]
        X_broadcast = tf.reshape(X, (batchSize, 1, dimensionSize))
        sumOfSquareDistances = tf.reduce_sum(tf.square(tf.subtract(X_broadcast, U)), 2)
        return sumOfSquareDistances

    def LnProbabilityXGivenZ(self, data, mean, variance):
        sumOfSquare = self.PairwiseDistances(data, mean)
        logLikelihoodDataGivenCluster = tf.add(-tf.multiply(tf.cast(self.D, tf.float32)/2.0,tf.log(tf.constant(2.0*np.pi)*variance)), -tf.divide(sumOfSquare, 2.0*variance))
        return logLikelihoodDataGivenCluster

    def LnProbabilityZGivenX(self, data, mean, variance, lnPriorBroad):
        lnProbabilityXGivenZ = self.LnProbabilityXGivenZ(data, mean, variance)
        # lnPriorBroad = tf.log(tf.reshape(prior, (1, self.K)))
        numerator = lnPriorBroad + lnProbabilityXGivenZ
        lnProbabilityX = tf.reshape(reduce_logsumexp(numerator, 1), (tf.shape(data)[0], 1))
        lnProbabilityZGivenX = numerator - lnProbabilityX
        return lnProbabilityZGivenX
        # Monotonically increasing, others doesnt matter ??
        # return numerator

    def LnProbabilityX(self, data, mean, variance, lnPriorBroad):
        lnProbabilityXGivenZ = self.LnProbabilityXGivenZ(data, mean, variance)
        # lnPriorBroad = tf.log(tf.reshape(prior, (1, self.K)))
        numerator = lnPriorBroad + lnProbabilityXGivenZ
        lnProbabilityX = tf.reshape(reduce_logsumexp(numerator, 1), (tf.shape(data)[0], 1))
        return lnProbabilityX

    def MixtureOfGaussiansMethod(self):
        ''' 
        Build Graph and execute in here
        so don't have to pass variables one by one
        Bad Coding Style but higher programmer productivity
        '''
        # Build Graph 
        # Mean location matters a lot in convergence
        clusterMean = tf.Variable(tf.truncated_normal([self.K, self.D], mean=-1, stddev=2.0)) # cluster centers
        clusterStdDeviationConstraint = tf.Variable(tf.truncated_normal([1, self.K], mean=0, stddev=0.1))
        clusterVariance = tf.exp(clusterStdDeviationConstraint)
        clusterStdDeviation = tf.sqrt(clusterVariance)
        # Uniform intialization
        clusterPriorConstraint = tf.Variable(tf.ones([1, self.K]))
        logClusterConstraint = logsoftmax(clusterPriorConstraint)
        clusterPrior = tf.exp(logClusterConstraint)

        trainData = tf.placeholder(tf.float32, shape=[None, self.D], name="trainingData")

        sumOfSquare = self.PairwiseDistances(trainData, clusterMean)
        lnProbabilityXGivenZ = self.LnProbabilityXGivenZ(trainData, clusterMean, clusterVariance)
        lnProbabilityX = self.LnProbabilityX(trainData, clusterMean, clusterVariance, logClusterConstraint)
        loss = (tf.reduce_sum(-1.0 * lnProbabilityX))
        # This is needed to decide which assignment it is
        lnProbabilityZGivenX = self.LnProbabilityZGivenX(trainData, clusterMean, clusterVariance, logClusterConstraint)
        probabilityZGivenX = tf.exp(lnProbabilityZGivenX)
        check = tf.reduce_sum(probabilityZGivenX, 1) # Check probabilities sum to 1
        # Assign classes based on maximum posterior probability for each data point
        minAssignments = tf.argmax(lnProbabilityXGivenZ, 1) # No prior contribution during assignment
        minAssignments = tf.argmax(lnProbabilityZGivenX, 1) # Prior contributes during assignment

        # ----------------------------------------------------------------------------------
        #logLikelihoodDataGivenCluster = self.LnProbabilityZGivenX(trainData, clusterMean, clusterStdDeviation, clusterPrior)
        minValidAssignments = minAssignments #Initialization
        if self.hasValid: 
            valid_data = tf.placeholder(tf.float32, shape=[None, self.D], name="validationData")
            validLoss = tf.reduce_sum(-1.0 * self.LnProbabilityX(valid_data, clusterMean,clusterVariance,logClusterConstraint))
            validLnProbabilityZGivenX = self.LnProbabilityZGivenX(valid_data, clusterMean, clusterVariance, logClusterConstraint)
            minValidAssignments = tf.argmax(validLnProbabilityZGivenX, 1) # Prior contributes during assignment

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
                feedDicts = {trainData: self.trainData[step*self.miniBatchSize:(step+1)*self.miniBatchSize]}
                if self.hasValid:
                    feedDicts = {trainData: self.trainData[step*self.miniBatchSize:(step+1)*self.miniBatchSize], valid_data: self.validData}
                _, minAssignTrain, paramClusterMean, paramClusterPrior, paramClusterStdDeviation, zGivenX, checkZGivenX, errTrain, errValid, minAssignValid = sess.run([train, minAssignments, clusterMean, clusterPrior, clusterStdDeviation, lnProbabilityZGivenX, check, loss, validLoss, minValidAssignments], feed_dict = feedDicts)
                xAxis.append(numUpdate)
                yTrainErr.append(errTrain)
                yValidErr.append(errValid)
                step += 1
                numUpdate += 1
            currEpoch += 1

            if currEpoch%100 == 0:
                logStdOut("e: " + str(currEpoch))
        # Calculate everything again without training
        feedDicts = {trainData: self.trainData}
        # No training, just gather data for valid assignments 
        if self.hasValid:
            feedDicts = {trainData: self.trainData, valid_data: self.validData}
        minAssignTrain, paramClusterMean, paramClusterPrior, paramClusterStdDeviation, zGivenX, checkZGivenX, errTrain, errValid, minAssignValid = sess.run([minAssignments, clusterMean, clusterPrior, clusterStdDeviation, lnProbabilityZGivenX, check, loss, validLoss, minValidAssignments], feed_dict = feedDicts)
        # Count how many assigned to each class
        currTrainDataShuffle = self.trainData
        self.printPlotResults(xAxis, yTrainErr, yValidErr, numUpdate, minAssignTrain, currTrainDataShuffle, paramClusterMean, paramClusterStdDeviation, paramClusterPrior, minAssignValid)

def executeMixtureOfGaussians(questionTitle, K, dataType, hasValid, numEpoch, learningRate):
    """
    Re-loads the data and re-randomize it with same seed anytime to ensure replicable results
    """
    logStdOut(questionTitle)
    print questionTitle
    trainData = 0
    validData = 0
    # Load data with seeded randomization
    dataInitializer = DataInitializer()
    if hasValid:
        trainData, validData = dataInitializer.getData(dataType, hasValid)
    else: 
        trainData = dataInitializer.getData(dataType, hasValid)

    # Execute algorithm 
    kObject = MixtureOfGaussians(questionTitle, K, trainData, validData, hasValid, dataType, numEpoch, learningRate)
    logElapsedTime(questionTitle + "K" + str(K) + "NumEpoch" + str(numEpoch))

# Global for logging
questionTitle = "" # Need to be global for logging to work
startTime = datetime.datetime.now()
figureCount = 1 # To not overwrite existing pictures

def logStdOut(message):
    # Temporary print to std out
    sys.stdout = sys.__stdout__
    print message
    # Continue editing same file
    sys.stdout = open("result" + questionTitle + ".txt", "a")

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
    print "ECE521 Assignment 3: Unsupervised Learning: GaussianCluster"
    '''
    # Gaussian Cluster Model
    questionTitle = "2.1.2" # Implemented function
    questionTitle = "2.1.3" # Implemented FUnction
    print "ECE521 Assignment 3: Unsupervised Learning: Mixture of Gaussian"
    questionTitle = "2.2.2"
    dataType = "2D"
    hasValid = False # No validation data
    K = 3
    numEpoch = 200
    learningRate = 0.1
    # Note: Loss will be higher since no validation data
    executeMixtureOfGaussians(questionTitle, K, dataType, hasValid, numEpoch, learningRate)
    # '''

    '''
    questionTitle = "2.2.3"
    dataType = "2D"
    hasValid = True
    diffK = [1, 2, 3, 4, 5]
    numEpoch = 200
    learningRate = 0.1
    for K in diffK:
        executeMixtureOfGaussians(questionTitle, K, dataType, hasValid, numEpoch, learningRate)
    # '''

    questionTitle = "2.2.4.2"
    dataType = "100D"
    hasValid = True
    diffK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    numEpoch = 150
    learningRate = 0.1
    for K in diffK:
        executeMixtureOfGaussians(questionTitle, K, dataType, hasValid, numEpoch, learningRate)
    # '''
