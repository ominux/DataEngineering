import tensorflow as tf
import numpy as np
import sys
from dataInitializer import DataInitializer
from utils import * 

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

    def printPlotResults(self, xAxis, yTrainErr, yValidErr, numUpdate, minAssign, currTrainData, numAssignEachClass, clusterMean, clusterStdDeviation, clusterPrior):
        # TODO: Print by decreasing so that the outliers get printed over the non-outliers
        figureCount = 0 # TODO: Make global
        import matplotlib.pyplot as plt

        print "mean", clusterMean
        print "K: ", self.K
        print "Iter: ", numUpdate
        print "Assignments To Classes:", numAssignEachClass
        percentageAssignEachClass = numAssignEachClass/float(sum(numAssignEachClass))
        print "Percentage Assignment To Classes:", percentageAssignEachClass
        print "prior", clusterPrior
        print "prior.shape", clusterPrior.shape
        print "prior Sum", np.sum(clusterPrior)
        print "stdDeviation", clusterStdDeviation
        print "stdDeviationShape", clusterStdDeviation.shape
        print "Lowest TrainLoss", np.min(yTrainErr)
        print "Lowest ValidLoss", np.min(yValidErr)

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
        figureCount = figureCount + 1
        plt.figure(figureCount)
        plt.axes()

        title = trainStr + typeScatterStr + paramStr
        plt.title(title)
        plt.xlabel(dimensionOneStr)
        plt.ylabel(dimensionTwoStr)
        colors = ['blue', 'red', 'green', 'black', 'yellow']
        plt.scatter(currTrainData[:, 0], currTrainData[:, 1], c=minAssign, s=10, alpha=0.5)
        colors = colors[:self.K]
        for i, j, k, l in zip(clusterMean, percentageAssignEachClass, colors, clusterStdDeviation[0]):
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
        #clusterMean = tf.Variable(tf.zeros([self.K, self.D])) # cluster centers
        # Initialize to [-1, 1]
        # clusterStdDeviationConstraint = tf.Variable(tf.truncated_normal([1, self.K], mean=-0.69, stddev = 0.3)) # cluster constraint to prevent negative
        #clusterStdDeviationConstraint = tf.Variable(tf.truncated_normal([1, self.K], mean=0, stddev = 0.5)) # cluster constraint to prevent negative
        clusterStdDeviationConstraint = tf.Variable(tf.truncated_normal([1, self.K], mean=0, stddev=0.1))
        #clusterStdDeviationConstraint = tf.Variable(tf.zeros([1, self.K]))
        clusterVariance = tf.exp(clusterStdDeviationConstraint)
        clusterStdDeviation = tf.sqrt(clusterVariance)
        # Uniform intialization
        #clusterPriorConstraint = tf.Variable(tf.ones([1, self.K]))
        clusterPriorConstraint = tf.ones([1, self.K])
        logClusterConstraint = logsoftmax(tf.log(clusterPriorConstraint))
        # clusterPrior = tf.divide(tf.exp(clusterPriorConstraint), tf.reduce_sum(tf.exp(clusterPriorConstraint)))
        clusterPrior = tf.exp(logClusterConstraint)

        trainData = tf.placeholder(tf.float32, shape=[None, self.D], name="trainingData")

        sumOfSquare = self.PairwiseDistances(trainData, clusterMean)
        lnProbabilityXGivenZ = self.LnProbabilityXGivenZ(trainData, clusterMean, clusterVariance)
        #lnProbabilityX = self.LnProbabilityX(trainData, clusterMean, clusterStdDeviation, clusterPrior)
        lnProbabilityX = self.LnProbabilityX(trainData, clusterMean, clusterVariance, logClusterConstraint)
        #loss = tf.negative(tf.reduce_sum(lnProbabilityX))
        loss = (tf.reduce_sum(-1.0 * lnProbabilityX))
        # This is needed to decide which assignment it is
        #lnProbabilityZGivenX = self.LnProbabilityZGivenX(trainData, clusterMean, clusterStdDeviation, clusterPrior)
        lnProbabilityZGivenX = self.LnProbabilityZGivenX(trainData, clusterMean, clusterVariance, logClusterConstraint)
        probabilityZGivenX = tf.exp(lnProbabilityZGivenX)
        check = tf.reduce_sum(probabilityZGivenX, 1) # Check probabilities sum to 1
        # Assign classes based on maximum posterior probability for each data point
        #minAssignments = tf.argmax(lnProbabilityZGivenX, 1)
        minAssignments = tf.argmax(lnProbabilityXGivenZ, 1)

        # ----------------------------------------------------------------------------------
        #logLikelihoodDataGivenCluster = self.LnProbabilityZGivenX(trainData, clusterMean, clusterStdDeviation, clusterPrior)
        validLoss = loss # initialization
        if self.hasValid: 
            valid_data = tf.placeholder(tf.float32, shape=[None, self.D], name="validationData")
            validLoss = tf.reduce_sum(-1.0 * self.LnProbabilityX(valid_data, clusterMean,clusterVariance,logClusterConstraint))

        train = self.optimizer.minimize(loss)
        
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
            step = 0
            while step*self.miniBatchSize < self.trainData.shape[0]:
                feedDicts = {trainData: self.trainData[step*self.miniBatchSize:(step+1)*self.miniBatchSize]}
                if self.hasValid:
                    feedDicts = {trainData: self.trainData[step*self.miniBatchSize:(step+1)*self.miniBatchSize], valid_data: self.validData}
                _, minAssign, paramClusterMean, paramClusterPrior, paramClusterStdDeviation, zGivenX, checkZGivenX, errTrain, errValid = sess.run([train, minAssignments, clusterMean, clusterPrior, clusterStdDeviation, lnProbabilityZGivenX, check, loss, validLoss], feed_dict = feedDicts)
                # print checkZGivenX
                '''
                print "prior", paramClusterPrior
                print "prior.shape", paramClusterPrior.shape
                print "prior Sum", np.sum(paramClusterPrior)
                print "stdDeviation", paramClusterStdDeviation
                print "stdDeviationShape", paramClusterStdDeviation.shape
                print "zgivenX", zGivenX
                print "zgivenXshape", zGivenX.shape
                print "minAssign", minAssign
                sys.exit(0)
                '''
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
            # print currEpoch
        # Calculate everything again without training
        feedDicts = {trainData: self.trainData}
        if self.hasValid:
            feedDicts = {trainData: self.trainData, valid_data: self.validData}
        minAssign, paramClusterMean, paramClusterPrior, paramClusterStdDeviation, zGivenX, checkZGivenX, errTrain, errValid = sess.run([minAssignments, clusterMean, clusterPrior, clusterStdDeviation, lnProbabilityZGivenX, check, loss, validLoss], feed_dict = feedDicts)
        # Count how many assigned to each class
        numAssignEachClass = np.bincount(minAssign)
        currTrainDataShuffle = self.trainData
        self.printPlotResults(xAxis, yTrainErr, yValidErr, numUpdate, minAssign, currTrainDataShuffle, numAssignEachClass, paramClusterMean, paramClusterStdDeviation, paramClusterPrior)

def executeMixtureOfGaussians(questionTitle, K, dataType, hasValid, numEpoch, learningRate):
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
    kObject = MixtureOfGaussians(questionTitle, K, trainData, validData, hasValid, dataType, numEpoch, learningRate)

if __name__ == "__main__":
    '''
    print "ECE521 Assignment 3: Unsupervised Learning: GaussianCluster"
    # Gaussian Cluster Model
    questionTitle = "2.1.2" # Implemented function
    questionTitle = "2.1.3" # Implemented FUnction
    print "ECE521 Assignment 3: Unsupervised Learning: Mixture of Gaussian"
    questionTitle = "2.2.2"
    dataType = "2D"
    hasValid = False # No validation data
    K = 3
    numEpoch = 1000
    learningRate = 0.001
    # Note: Loss will be higher since no validation data
    executeMixtureOfGaussians(questionTitle, K, dataType, hasValid, numEpoch, learningRate)
    # '''

    questionTitle = "2.2.3"
    dataType = "2D"
    hasValid = True
    diffK = [1, 2, 3, 4, 5]
    diffK = [5]
    numEpoch = 1000
    learningRate = 0.01
    for K in diffK:
        executeMixtureOfGaussians(questionTitle, K, dataType, hasValid, numEpoch, learningRate)
    # '''

    '''
    questionTitle = "2.2.4"
    dataType = "100D"
    hasValid = True
    diffK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    diffK = [10]
    numEpoch = 30 
    learningRate = 0.01
    for K in diffK:
        executeMixtureOfGaussians(questionTitle, K, dataType, hasValid, numEpoch, learningRate)
    # '''
