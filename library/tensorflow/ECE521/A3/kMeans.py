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
        # self.trainData = trainData[0:8] # TODO: TEMP DEBUG
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

    def printPlotResults(self, xAxis, yTrainErr, numUpdate):
        # TODO: To print graphs
        figureCount = 0 # TODO: Make global
        import matplotlib.pyplot as plt

        print "Iter: ", numUpdate

        trainStr = "Train"
        validStr = "Valid"
        typeLossStr = "Loss"
        trainLossStr = trainStr + typeLossStr
        iterationStr = "Iteration"
        paramStr = "LearninRate" + str(self.learningRate) + "NumEpoch" + str(self.numEpoch)

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
        minSquare = tf.reduce_min(sumOfSquare, 1) # TEMP
        loss = tf.reduce_sum(minSquare)

        train = self.optimizer.minimize(loss)

        
        # Session
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)
        currEpoch = 0
        xAxis = []
        yTrainErr = []
        yValidErr = []
        numUpdate = 0
        step = 0
        while currEpoch < self.numEpoch:
            np.random.shuffle(self.trainData) # Shuffle Batches
            step = 0
            feedDicts = {train_data: self.trainData}
            if self.hasValid:
                feedDicts = {train_data: self.trainData, valid_data: self.validData}
            while step*self.miniBatchSize < self.trainData.shape[0]:
                _, errTrain = sess.run([train, loss], feed_dict = feedDicts)
                '''
                # TEMP DEBUG, uncomment above once done
                _, errTrain, u, d, sqr, sumSqr, minSqr, x = sess.run([train, loss, U, deduct, square, sumOfSquare, minSquare, train_data_broad], feed_dict = feedDicts)
                print 'x', x
                print 'u', u
                print 'd', d
                print 'sqr', sqr
                print 'sumSqr', sumSqr
                print 'minSqr', minSqr
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
        self.printPlotResults(xAxis, yTrainErr, numUpdate)

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

