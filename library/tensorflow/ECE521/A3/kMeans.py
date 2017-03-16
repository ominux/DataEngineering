import tensorflow as tf
import numpy as np
from dataInitializer import DataInitializer

class KMeans(object):
    def __init__(self, K, trainData, validData, hasValid, numEpoch = 300, learningRate = 0.01, questionTitle = ""):
        """
        Constructor
        """
        self.K = K
        self.trainData = trainData
        self.validData = validData 
        #self.trainData = trainData[0:100] # TODO: TEMP DEBUG
        #numEpoch = 1
        self.D = self.trainData[0].size # Dimension of each data
        self.hasValid = hasValid
        self.learningRate = learningRate
        self.numEpoch = numEpoch
        self.miniBatchSize = self.trainData.shape[0] # TODO: Implement this or jsust entire gradient descent?
        print 'minibatchsize', self.miniBatchSize
        self.questionTitle = questionTitle
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate, beta1=0.9, beta2=0.99, epsilon=1e-5)
        # Execute KMeans
        self.KMeansMethod()

    def printPlotResults(self, xAxis, yTrainErr, numUpdate, minAssign, currTrainData):
        # TODO: To print graphs
        figureCount = 0 # TODO: Make global
        import matplotlib.pyplot as plt

        print "Iter: ", numUpdate

        trainStr = "Train"
        validStr = "Valid"
        typeLossStr = "Loss"
        typeScatterStr = "Assignments"
        trainLossStr = trainStr + typeLossStr
        iterationStr = "Iteration"
        dimensionOneStr = "D1"
        dimensionTwoStr = "D2"
        paramStr = "LearninRate" + str(self.learningRate) + "NumEpoch" + str(self.numEpoch)

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

        # Plot percentage in each different classes as well
        # TODO: Plot percentage as part of it as well
        # 2.1.3
        # all the trainData dimensions after assigning into the different K's

        # Scatter plot based on assignment colors
        figureCount = figureCount + 1
        plt.figure(figureCount)
        title = trainStr + typeScatterStr + paramStr
        plt.title(title)
        plt.xlabel(dimensionOneStr)
        plt.ylabel(dimensionTwoStr)
        k = 0
        # TODO: Loop through the different classes and scatter one by one
        plt.scatter(currTrainData[:, 0], currTrainData[:, 1], c=minAssign, s=50, alpha=0.5)
        plt.savefig(self.questionTitle + title + ".png")
        plt.close()
        plt.clf()
        # '''


    def KMeansMethod(self):
        ''' 
        Build Graph and execute in here
        so don't have to pass variables one by one
        Bad Coding Style but higher programmer productivity
        '''
        # Build Graph 
        U = tf.Variable(tf.truncated_normal([self.K, self.D])) # TODO: Set own std. deviation ?

        train_data = tf.placeholder(tf.float32, shape=[None, self.D], name="trainingData")
        batchSizing = tf.shape(train_data)[0]

        if self.hasValid: 
            valid_data = tf.placeholder(tf.float32, shape=[None, self.D], name="validationData")

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
                _, minAssign, errTrain = sess.run([train, minAssignments, loss], feed_dict = feedDicts)
                # TEMP DEBUG, uncomment above once done
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
                step += 1
                numUpdate += 1
            currEpoch += 1
            if currEpoch%50 == 0:
                print("e", str(currEpoch))
        # TODO: Print Plots
        self.printPlotResults(xAxis, yTrainErr, numUpdate, minAssign, currTrainDataShuffle)

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
    print "K:", K
    kObject = KMeans(K, trainData, validData, hasValid)

if __name__ == "__main__":
    print "ECE521 Assignment 3: Unsupervised Learning: K Means"
    # Unsupervised => Data has no label or target
    questionTitle = "1.1.2"
    dataType = "2D"
    hasValid = False # No validation data
    K = 3
    executeKMeans(questionTitle, K, dataType, hasValid)
    # '''

    '''
    questionTitle = "1.1.3"
    diffK = [1 2 3 4 5]
    dataType = "2D"
    hasValid = False
    for K in diffK:
        executeKMeans(questionTitle, K, dataType, hasValid)
    # '''

    '''
    questionTitle = "1.1.4"
    diffK = [1 2 3 4 5]
    dataType = "2D"
    hasValid = True
    for K in diffK:
        executeKMeans(questionTitle, K, dataType, hasValid)
    # '''

