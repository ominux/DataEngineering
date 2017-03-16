import tensorflow as tf
import numpy as np
from dataInitializer import DataInitializer

class KMeans(object):
    def __init__(self, questionTitle, K, trainData, validData, hasValid, numEpoch = 100, learningRate = 0.1):
        """
        Constructor
        """
        self.K = K
        self.trainData = trainData
        self.validData = validData 
        #self.trainData = trainData[0:100] # TODO: TEMP DEBUG
        #numEpoch = 1
        #numEpoch = 50
        self.D = self.trainData[0].size # Dimension of each data
        self.hasValid = hasValid
        self.learningRate = learningRate
        self.numEpoch = numEpoch
        self.miniBatchSize = self.trainData.shape[0] # miniBatchSize is entire data size
        self.questionTitle = questionTitle
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate, beta1=0.9, beta2=0.99, epsilon=1e-5)
        # Execute KMeans
        self.KMeansMethod()

    def printPlotResults(self, xAxis, yTrainErr, yValidErr, numUpdate, minAssign, currTrainData, numAssignEachClass):
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
        # TODO: Fix this to plot label properly with colors or at least know what colors represents which
        plt.scatter(currTrainData[:, 0], currTrainData[:, 1], c=minAssign, s=50, alpha=0.5, label=percentageAssignEachClass)
        plt.legend()
        plt.savefig(self.questionTitle + title + ".png")
        plt.close()
        plt.clf()


    def KMeansMethod(self):
        ''' 
        Build Graph and execute in here
        so don't have to pass variables one by one
        Bad Coding Style but higher programmer productivity
        '''
        # Build Graph 
        U = tf.Variable(tf.truncated_normal([self.K, self.D]))

        train_data = tf.placeholder(tf.float32, shape=[None, self.D], name="trainingData")
        batchSizing = tf.shape(train_data)[0]

        if self.hasValid: 
            valid_data = tf.placeholder(tf.float32, shape=[None, self.D], name="validationData")
            validBatchSizing = tf.shape(valid_data)[0]
            valid_data_broad = tf.reshape(valid_data, (validBatchSizing, 1, self.D))
            validLoss = tf.reduce_sum(tf.reduce_min(tf.reduce_sum(tf.square(valid_data_broad-U), 2), 1))

        train_data_broad = tf.reshape(train_data, (batchSizing, 1, self.D))
        deduct = train_data_broad - U
        square = tf.square(deduct)
        sumOfSquare =  tf.reduce_sum(square, 2)
        minSquare = tf.reduce_min(sumOfSquare, 1)
        loss = tf.reduce_sum(minSquare)

        train = self.optimizer.minimize(loss)

        minAssignments = tf.argmin(sumOfSquare,1)
        
        # Session
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)
        currEpoch = 0
        minAssign = 0
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
            feedDicts = {train_data: self.trainData}
            if self.hasValid:
                feedDicts = {train_data: self.trainData, valid_data: self.validData}
            while step*self.miniBatchSize < self.trainData.shape[0]:
                _, minAssign, errTrain, errValid = sess.run([train, minAssignments, loss, validLoss], feed_dict = feedDicts)
                #_, minAssign, errTrain = sess.run([train, minAssignments, loss], feed_dict = feedDicts)
                # TODO: TEMP DEBUG, uncomment above once done
                '''
                _, errTrain, u, d, sqr, sumSqr, minSqr, minAssign, x = sess.run([train, loss, U, deduct, square, sumOfSquare, minSquare, minAssignments, train_data_broad], feed_dict = feedDicts)
                print 'x', x
                print 'u', u
                print 'd', d
                print 'sqr', sqr
                print 'sumSqr', sumSqr
                print 'minSqr', minSqr
                print 'minAssign', minAssign
                print 'l', errTrain
                # '''
                xAxis.append(numUpdate)
                yTrainErr.append(errTrain)
                yValidErr.append(errValid)
                step += 1
                numUpdate += 1
            currEpoch += 1
            if currEpoch%50 == 0:
                doNothing = 0
                # print("e", str(currEpoch))
        # Count how many assigned to each class
        numAssignEachClass = np.bincount(minAssign)
        self.printPlotResults(xAxis, yTrainErr, yValidErr, numUpdate, minAssign, currTrainDataShuffle, numAssignEachClass)

def executeKMeans(questionTitle, K, dataType, hasValid):
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
    kObject = KMeans(questionTitle, K, trainData, validData, hasValid)

if __name__ == "__main__":
    print "ECE521 Assignment 3: Unsupervised Learning: K Means"
    '''
    # Unsupervised => Data has no label or target
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

    questionTitle = "1.1.4"
    diffK = [1, 2, 3, 4, 5]
    dataType = "2D"
    hasValid = True
    for K in diffK:
        executeKMeans(questionTitle, K, dataType, hasValid)
    # '''
