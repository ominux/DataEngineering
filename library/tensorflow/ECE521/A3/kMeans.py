import tensorflow as tf
import numpy as np
from dataInitializer import DataInitializer
import datetime
import sys

class KMeans(object):
    def __init__(self, questionTitle, K, trainData, validData, hasValid, dataType, numEpoch = 50, learningRate = 0.1):
        """
        Constructor
        """
        self.K = K
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
        # Execute KMeans
        self.KMeansMethod()

    def printPlotResults(self, xAxis, yTrainErr, yValidErr, numUpdate, minAssignTrain, currTrainData, centers, minAssignValid):
        figureCount = 0 # TODO: Make global
        import matplotlib.pyplot as plt

        print "K: ", self.K
        print "Iter: ", numUpdate
        print str(self.K) + "Lowest TrainLoss", np.min(yTrainErr)
        print str(self.K) + "Lowest ValidLoss", np.min(yValidErr)
        # Count how many assigned to each class
        numTrainAssignEachClass = np.bincount(minAssignTrain)
        numValidAssignEachClass = np.bincount(minAssignValid)
        print "Train Percentage Assignment To Classes:", percentageTrainAssignEachClass
        print "Train Assignments To Classes:", numTrainAssignEachClass
        percentageTrainAssignEachClass = numTrainAssignEachClass/float(sum(numTrainAssignEachClass))

        percentageValidAssignEachClass = percentageTrainAssignEachClass # Initialize

        if self.hasValid:
            print "Valid Assignments To Classes:", numValidAssignEachClass
            percentageValidAssignEachClass = numValidAssignEachClass/float(sum(numValidAssignEachClass))
            print "Valid Percentage Assignment To Classes:", percentageValidAssignEachClass

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
        title = trainStr + typeScatterStr + paramStr
        plt.title(title)
        plt.xlabel(dimensionOneStr)
        plt.ylabel(dimensionTwoStr)
        colors = ['blue', 'red', 'green', 'black', 'yellow', 'magenta', 'cyan', 'brown', 'orange', 
                'aqua']
        colors = colors[:self.K]
        plt.scatter(currTrainData[:, 0], currTrainData[:, 1], c=minAssignTrain, s=10, alpha=0.5)
        for i, j, k in zip(centers, percentageTrainAssignEachClass, colors):
            plt.plot(i[0], i[1], 'kx', markersize=15, label=j, c=k)
        plt.legend()
        plt.savefig(self.questionTitle + title + ".png")
        plt.close()
        plt.clf()

        if self.hasValid:
            # Valid Assignments
            figureCount = figureCount + 1
            plt.figure(figureCount)
            title = validStr + typeScatterStr + paramStr
            plt.title(title)
            plt.xlabel(dimensionOneStr)
            plt.ylabel(dimensionTwoStr)
            colors = ['blue', 'red', 'green', 'black', 'yellow', 'magenta', 'cyan', 'brown', 'orange', 
                    'aqua']
            colors = colors[:self.K]
            plt.scatter(self.validData[:, 0], self.validData[:, 1], c=minAssignValid, s=10, alpha=0.5)
            for i, j, k in zip(centers, percentageValidAssignEachClass, colors):
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
            Distances = matrix of size (B x D) containing the pairwise Euclidean distances
        """
        batchSize = tf.shape(X)[0] 
        dimensionSize = tf.shape(X)[1]
        numClusters = tf.shape(U)[0]
        X_broadcast = tf.reshape(X, (batchSize, 1, dimensionSize))
        sumOfSquareDistances = tf.reduce_sum(tf.square(tf.subtract(X_broadcast, U)), 2)
        return sumOfSquareDistances

    def KMeansMethod(self):
        ''' 
        Build Graph and execute in here
        so don't have to pass variables one by one
        Bad Coding Style but higher programmer productivity
        '''
        # Build Graph 
        U = tf.Variable(tf.truncated_normal([self.K, self.D]))
        train_data = tf.placeholder(tf.float32, shape=[None, self.D], name="trainingData")
        sumOfSquare = self.PairwiseDistances(train_data, U)
        minSquare = tf.reduce_min(sumOfSquare, 1)
        minAssignments = tf.argmin(sumOfSquare,1)
        loss = tf.reduce_sum(minSquare)
        validLoss = loss
        minValidAssignments = minAssignments

        if self.hasValid: 
            valid_data = tf.placeholder(tf.float32, shape=[None, self.D], name="validationData")
            validSumOfSquare = self.PairwiseDistances(valid_data, U)
            validLoss = tf.reduce_sum(tf.reduce_min(validSumOfSquare, 1))
            minValidAssignments = tf.argmin(validSumOfSquare, 1)

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
            currTrainDataShuffle = self.trainData
            step = 0
            while step*self.miniBatchSize < self.trainData.shape[0]:
                feedDicts = {train_data: self.trainData[step*self.miniBatchSize:(step+1)*self.miniBatchSize]}
                if self.hasValid:
                    feedDicts = {train_data: self.trainData[step*self.miniBatchSize:(step+1)*self.miniBatchSize], valid_data:self.validData}
                _, minAssignTrain, minAssignValid, centers, errTrain, errValid = sess.run([train, minAssignments, minValidAssignments, U, loss, validLoss], feed_dict = feedDicts)
                xAxis.append(numUpdate)
                yTrainErr.append(errTrain)
                yValidErr.append(errValid)
                step += 1
                numUpdate += 1
            currEpoch += 1
            if currEpoch%50 == 0:
                logStdOut("e: " + str(currEpoch))
        if self.dataType == "2D":
            print "Center Values", centers
        self.printPlotResults(xAxis, yTrainErr, yValidErr, numUpdate, minAssignTrain, currTrainDataShuffle, centers, minAssignValid)

def executeKMeans(questionTitle, K, dataType, hasValid):
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
    kObject = KMeans(questionTitle, K, trainData, validData, hasValid, dataType)
    logElapsedTime(questionTitle + "K" + str(K))
    

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
    print "ECE521 Assignment 3: Unsupervised Learning: K Means"

    # Unsupervised => Data has no label or target
    '''
    questionTitle = "1.1.2"
    dataType = "2D"
    hasValid = False # No validation data
    K = 3
    executeKMeans(questionTitle, K, dataType, hasValid)
    # '''

    '''
    questionTitle = "1.1.3"
    diffK = [1, 2, 3, 4, 5]
    dataType = "2D"
    hasValid = False
    for K in diffK:
        executeKMeans(questionTitle, K, dataType, hasValid)
    # '''

    '''
    questionTitle = "1.1.4"
    diffK = [1, 2, 3, 4, 5]
    dataType = "2D"
    hasValid = True
    for K in diffK:
        executeKMeans(questionTitle, K, dataType, hasValid)
    # '''

    # Run using 100D data
    questionTitle = "2.2.4.1"
    diffK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dataType = "100D"
    hasValid = True
    for K in diffK:
        executeKMeans(questionTitle, K, dataType, hasValid)
    # '''
