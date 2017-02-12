import numpy as np
import tensorflow as tf
import sys

# TODO: Replace all 784 with some automatic calculation
class LogisticRegression(object):
    def __init__(self, trainData, trainTarget, validData, validTarget, testData, testTarget, learningRate = 0.0001):
        self.trainData = trainData
        self.trainTarget = trainTarget
        self.validData = validData
        self.validTarget = validTarget
        self.testData = testData
        self.testTarget = testTarget
        # Default hyperparameter values
        self.weightDecay = 0.01
        self.miniBatchSize = 500
        # MeanSquareError learningRate = 0.001, otherwise overshoots 
        # CrossEntropyError, learningRate = 0.01, 98.6% test accuracy highest
        self.learningRate = learningRate
        self.numEpoch = 5000
        self.numEpoch = 200

    # Logistic Regression 
    def LogisticRegressionMethod(self):
        """
        Implements logistic regression and cross-entropy loss.
        using tf.nn.sigmoid_cross_etnropy_with_logits
        Weight-decay coefficient = 0.01
        Mini-batch size = B = 500
        Two Class notMNIST dataset

        Output:
            Plots best training curve
                Cross-entropy Loss vs Number of updates
                Classification Accuracy vs Number of Updates
            Plots best test curve
                Cross-entropy Loss vs Number of updates
                Classification Accuracy vs Number of Updates
            Best Test Classification Accuracy
        """

        maxTestClassificationAccuracy = 0.0
        W, b, X, y_target, y_predicted, crossEntropyError, train , needTrain, accuracy = self.buildGraph()
        figureCount = 1 

        # Session
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)
        initialW = sess.run(W)  
        initialb = sess.run(b)
        currEpoch = 0
        wList = []
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
                _, errTrain, currentW, currentb, yhat, accTrain= sess.run([train,crossEntropyError, W, b, y_predicted, accuracy], feed_dict={X: np.reshape(self.trainData[step*self.miniBatchSize:(step+1)*self.miniBatchSize], (self.miniBatchSize,784)),y_target: self.trainTarget[step*self.miniBatchSize:(step+1)*self.miniBatchSize], needTrain: True})
                wList.append(currentW)
                step = step + 1
                xAxis.append(numUpdate)
                numUpdate += 1
                yTrainErr.append(errTrain)
                # These will not optimize the function cause you did not fetch 'train' 
                # So it won't have to execute that.
                errValid, accValid = sess.run([crossEntropyError, accuracy], feed_dict={X: np.reshape(self.validData, (self.validData.shape[0],784)), y_target: self.validTarget, needTrain: False})

                errTest, accTest = sess.run([crossEntropyError, accuracy], feed_dict={X: np.reshape(self.testData, (self.testData.shape[0], 784)), y_target: self.testTarget, needTrain: False})
                yValidErr.append(errValid)
                yTestErr.append(errTest)
                yValidAcc.append(accValid)
                yTestAcc.append(accTest)

                yTrainAcc.append(accTrain)
            currEpoch += 1
        print "LearningRate: " , self.learningRate, " Mini batch Size: ", self.miniBatchSize
        print "Iter: ", numUpdate
        print "Final Train MSE: ", errTrain
        print "Final Valid MSE: ", errValid
        print "Final Test MSE: ", errTest
        print "Final Train Acc: ", accTrain
        print "Final Valid Acc: ", accValid
        print "Final Test Acc: ", accTest
        import matplotlib.pyplot as plt
        plt.figure(figureCount)
        figureCount = figureCount + 1
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

    # Build the computational graph
    def buildGraph(self):
        # Parameters to train
        # Images are 28*28 = 784 pixels
        W = tf.Variable(tf.truncated_normal(shape=[784, 1], stddev=0.5), name='weights')
        b = tf.Variable(0.0, name='biases')

        # Supervised Inputs
        X = tf.placeholder(tf.float32, [None, 784], name='input_x')
        y_target = tf.placeholder(tf.float32, [None,1], name='target_y')

        # Label to know if it should train or simply return the errors
        needTrain = tf.placeholder(tf.bool)

        weightDecayCoeff = tf.div(tf.constant(self.weightDecay),tf.constant(2.0))
        # Graph definition
        y_predicted = tf.matmul(X, W) + b

        correctPred = tf.equal(tf.cast(tf.greater_equal(y_predicted, 0.5), tf.float32), tf.floor(y_target))
        accuracy = tf.reduce_mean(tf.cast(correctPred, "float"))

        # Weight Decay Error calculation
        weightDecayMeanSquareError = tf.reduce_mean(tf.square(W))

        weightDecayError = tf.multiply(weightDecayCoeff, weightDecayMeanSquareError)

        # Mean Square Error Calculation
        # Divide by 2M instead of M
        meanSquaredError = tf.div(tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target), 
                                                    reduction_indices=1, 
                                                    name='squared_error'), 
                                      name='mean_squared_error'), tf.constant(2.0))
        meanSquaredError = tf.add(meanSquaredError, weightDecayError)

        # Cross Entropy Error Calculation
        crossEntropyError = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_predicted, y_target))
        crossEntropyError = tf.add(crossEntropyError, weightDecayError)

        # Don't train if don't have to for validation and test set
        finalTrainingError = tf.select(needTrain, crossEntropyError, tf.constant(0.0))

        # Training mechanism
        gdOptimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learningRate)
        adamOptimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate)
        
        # Train and update the parameters defined
        # sess.run(train) will execute the optimized function
        # train = optimizer.minimize(loss=finalTrainingError)
        train = adamOptimizer.minimize(loss=finalTrainingError)

        # TODO: Return both errors for plotting
        return W, b, X, y_target, y_predicted, crossEntropyError, train, needTrain, accuracy

    def ShuffleBatches(self, trainData, trainTarget):
        # Gets the state as the current time
        rngState = np.random.get_state()
        np.random.shuffle(trainData)
        np.random.set_state(rngState)
        np.random.shuffle(trainTarget)
        return trainData, trainTarget

if __name__ == "__main__":
    print "LogisticRegression"
    # Binary Classification
    # Get only 2 labels
    with np.load("notMNIST.npz") as data :
        Data, Target = data ["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
        for learningRate in [0.01]:
            tf.reset_default_graph()
            l = LogisticRegression(trainData, trainTarget, validData, validTarget, testData, testTarget, learningRate)
            maxTestAccuracy = l.LogisticRegressionMethod()
            print "Max Test Accuracy is: ", maxTestAccuracy
    ''' 
    # Multi-class Classification
    # Get all 10 labels
    with np.load("notMNIST.npz") as data:
        Data, Target = data ["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    '''
