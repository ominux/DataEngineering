import tensorflow as tf
import numpy as np
import sys
from dataInitializer import DataInitializer
from utils import * 

class MixtureOfGaussians(object):
    def __init__(self, questionTitle, K, trainData, validData, hasValid, numEpoch = 200, learningRate = 0.1):
        """
        Constructor
        """
        self.K = K # number of clusters
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

    def printPlotResults(self, xAxis, yTrainErr, yValidErr, numUpdate, minAssign, currTrainData, numAssignEachClass, centers):
        figureCount = 0 # TODO: Make global
        import matplotlib.pyplot as plt

        print "K: ", self.K
        print "Iter: ", numUpdate
        print "Assignments To Classes:", numAssignEachClass
        percentageAssignEachClass = numAssignEachClass/float(sum(numAssignEachClass))
        print "Percentage Assignment To Classes:", percentageAssignEachClass

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

        # Plot percentage in each different classes as well
        # Scatter plot based on assignment colors
        # Including percentage as the label
        figureCount = figureCount + 1
        plt.figure(figureCount)
        title = trainStr + typeScatterStr + paramStr
        plt.title(title)
        plt.xlabel(dimensionOneStr)
        plt.ylabel(dimensionTwoStr)
        plt.scatter(currTrainData[:, 0], currTrainData[:, 1], c=minAssign, s=50, alpha=0.5)
        colors = ['blue', 'red', 'green', 'black', 'yellow']
        colors = colors[:self.K]
        for i, j, k in zip(centers, percentageAssignEachClass, colors):
            plt.plot(i[0], i[1], 'kx', markersize=15, label=j, c=k)
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

    def LnProbabilityXGivenZ(self, data, mean, stddev):
        sumOfSquare = self.PairwiseDistances(data, mean)
        logLikelihoodDataGivenCluster = -tf.log(tf.sqrt(tf.constant(2*np.pi)) * stddev) - tf.divide(sumOfSquare, 2*tf.square(stddev))
        return logLikelihoodDataGivenCluster

    def LnProbabilityZGivenX(self, data, mean, stddev, prior):
        lnProbabilityXGivenZ = self.LnProbabilityXGivenZ(data, mean, stddev)
        lnPriorBroad = tf.log(tf.reshape(prior, (1, self.K)))
        numerator = lnPriorBroad + lnProbabilityXGivenZ
        lnProbabilityX = tf.reshape(reduce_logsumexp(numerator, 1), (tf.shape(data)[0], 1))
        lnProbabilityZGivenX = numerator - lnProbabilityX
        return lnProbabilityZGivenX

    def LnProbabilityX(self, data, mean, stddev, prior):
        lnProbabilityXGivenZ = self.LnProbabilityXGivenZ(data, mean, stddev)
        lnPriorBroad = tf.log(tf.reshape(prior, (1, self.K)))
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
        clusterMean = tf.Variable(tf.truncated_normal([self.K, self.D])) # cluster centers
        clusterStdDeviationConstraint = tf.Variable(tf.truncated_normal([self.K])) # cluster constraint to prevent negative
        clusterStdDeviation = tf.sqrt(tf.exp(clusterStdDeviationConstraint))
        clusterPriorConstraint = tf.Variable(tf.truncated_normal([self.K]))
        clusterPrior = tf.divide(tf.exp(clusterPriorConstraint), tf.reduce_sum(tf.exp(clusterPriorConstraint)))

        trainData = tf.placeholder(tf.float32, shape=[None, self.D], name="trainingData")

        sumOfSquare = self.PairwiseDistances(trainData, clusterMean)
        lnProbabilityXGivenZ = self.LnProbabilityXGivenZ(trainData, clusterMean, clusterStdDeviation)
        # TODO: Copy paste into functions
        # ----------------------------------------------------------------------------------
        lnProbabilityZGivenX = self.LnProbabilityZGivenX(trainData, clusterMean, clusterStdDeviation, clusterPrior)
        check = tf.reduce_sum(tf.exp(lnProbabilityZGivenX), 1) # Check probabilities sum to 1
        lnProbabilityX = self.LnProbabilityX(trainData, clusterMean, clusterStdDeviation, clusterPrior)

        # ----------------------------------------------------------------------------------
        #logLikelihoodDataGivenCluster = self.LnProbabilityZGivenX(trainData, clusterMean, clusterStdDeviation, clusterPrior)
        loss = tf.reduce_sum(-1 * lnProbabilityX)
        validLoss = loss # initialization
        if self.hasValid: 
            valid_data = tf.placeholder(tf.float32, shape=[None, self.D], name="validationData")
            '''
            validBatchSizing = tf.shape(valid_data)[0]
            valid_data_broad = tf.reshape(valid_data, (validBatchSizing, 1, self.D))
            validLoss = tf.reduce_sum(tf.reduce_min(tf.reduce_sum(tf.square(tf.subtract(valid_data_broad,clusterMean)), 2), 1))
            '''
            validLoss = tf.reduce_sum(-1 * self.LnProbabilityX(valid_data, clusterMean, clusterStdDeviation, clusterPrior))

        train = self.optimizer.minimize(loss)

        # TODO: Update assignment function below for MOG
        minAssignments = tf.argmin(sumOfSquare,1)
        
        # Session
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)
        currEpoch = 0
        minAssign = 0
        centers = 0
        xAxis = []
        yTrainErr = []
        yValidErr = []
        numUpdate = 0
        step = 0
        currTrainDataShuffle = self.trainData
        while currEpoch < self.numEpoch:
            np.random.shuffle(self.trainData) # Shuffle Batches
            currTrainDataShuffle = self.trainData
            step = 0
            while step*self.miniBatchSize < self.trainData.shape[0]:
                feedDicts = {trainData: self.trainData[step*self.miniBatchSize:(step+1)*self.miniBatchSize]}
                if self.hasValid:
                    feedDicts = {trainData: self.trainData[step*self.miniBatchSize:(step+1)*self.miniBatchSize], valid_data: self.validData}
                _, minAssign, centers, zGivenX, checkZGivenX, errTrain, errValid = sess.run([train, minAssignments, clusterMean, lnProbabilityZGivenX, check, loss, validLoss], feed_dict = feedDicts)
                # print checkZGivenX
                '''
                _, minAssign, centers, xGivenZ, clusterPri, num,den, zGivenX, errTrain, errValid = sess.run([train, minAssignments, clusterMean, lnProbabilityXGivenZ , clusterPrior, numerator, denominator, lnProbabilityZGivenX, loss, validLoss], feed_dict = feedDicts)
                print "lnXGivenZ", xGivenZ
                print xGivenZ.shape
                print "ClusterPrior", clusterPri
                print "numerator", num
                print "denominator", den
                print "lnZGivenX", zGivenX
                print zGivenX.shape
                sys.exit(0)
                '''
                xAxis.append(numUpdate)
                yTrainErr.append(errTrain)
                yValidErr.append(errValid)
                step += 1
                numUpdate += 1
            currEpoch += 1
        # Count how many assigned to each class
        numAssignEachClass = np.bincount(minAssign)
        self.printPlotResults(xAxis, yTrainErr, yValidErr, numUpdate, minAssign, currTrainDataShuffle, numAssignEachClass, centers)

def executeMixtureOfGaussians(questionTitle, K, dataType, hasValid):
    """
    Re-loads the data and re-randomize it with same seed anytime to ensure replicable results
    """
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
    kObject = MixtureOfGaussians(questionTitle, K, trainData, validData, hasValid)


if __name__ == "__main__":
    print "ECE521 Assignment 3: Unsupervised Learning: GaussianCluster"
    # Gaussian Cluster Model
    questionTitle = "2.1.2" # Implemented function
    questionTitle = "2.1.3" # Implemented FUnction
    print "ECE521 Assignment 3: Unsupervised Learning: Mixture of Gaussian"
    questionTitle = "2.2.2"
    dataType = "2D"
    hasValid = False # No validation data
    K = 3
    executeMixtureOfGaussians(questionTitle, K, dataType, hasValid)
    # '''

    '''
    # TODO:
    questionTitle = "2.2.3"
    for K in diffK:
        executeMixtureOfGaussians(questionTitle, K, dataType, hasValid)
    # TODO:
    questionTitle = "2.2.4"
    # TODO:
    # '''
