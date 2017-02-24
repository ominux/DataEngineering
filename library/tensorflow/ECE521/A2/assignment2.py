import numpy as np
import tensorflow as tf
import sys

# TODO: Replace all 784 with some automatic calculation for pixel size
class LogisticRegression(object):
    def __init__(self, trainData, trainTarget, validData, validTarget, testData, testTarget, learningRate = 0.0001):
        self.trainData = trainData
        self.trainTarget = trainTarget
        self.validData = validData
        self.validTarget = validTarget
        self.testData = testData
        self.testTarget = testTarget
        # For 1.1.3, weight decay is 0
        self.weightDecay = 0.00
        # For 1.1.3, should use entire batch instead of miniBatchSize, and already hard-coded as 3500 when fed in
        self.miniBatchSize = 500
        # Default hyperparameter values
        self.weightDecay = 0.01
        # MeanSquareError learningRate = 0.001, otherwise overshoots 
        # CrossEntropyError, learningRate = 0.01, 98.6% test accuracy highest
        # CrossEntropySoftmax Error, learningRate = 0.001
        self.learningRate = learningRate
        self.numEpoch = 0 
        self.numEpochBinary = 200
        # Normal Equation method really slow
        self.numEpochBinary = 1
        self.numEpochMulti = 100 # Multi-class Classification
        # self.numEpoch = 5000

    # Logistic Regression 
    def LogisticRegressionMethodBinary(self):
        """
        Implements logistic regression and cross-entropy loss.
        using tf.nn.sigmoid_cross_entropy_with_logits
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
        self.numEpoch = self.numEpochBinary
        W, b, X, y_target, y_predicted, crossEntropyError, train , needTrain, accuracy, WNormalEquation, accuracyNormal, Xall, y_targetAll = self.buildGraphBinary()
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
        yTrainNormalAcc = []
        yValidNormalAcc = []
        yTestNormalAcc = []
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
                _, errTrain, currentW, currentb, yhat, accTrain, currentWNormalEquation, accTrainNormal= sess.run([train, crossEntropyError, W, b, y_predicted, accuracy, WNormalEquation, accuracyNormal], feed_dict={X: np.reshape(self.trainData[step*self.miniBatchSize:(step+1)*self.miniBatchSize], (self.miniBatchSize,784)),Xall: np.reshape(self.trainData,(3500,784)), y_targetAll: self.trainTarget, y_target: self.trainTarget[step*self.miniBatchSize:(step+1)*self.miniBatchSize], needTrain: True})
                wList.append(currentW)
                step = step + 1
                xAxis.append(numUpdate)
                numUpdate += 1
                yTrainErr.append(errTrain)

                yTrainAcc.append(accTrain)
                yTrainNormalAcc.append(accTrainNormal)

                # These will not optimize the function cause you did not fetch 'train' 
                # So it won't have to execute that.
                errValid, accValid, accValidNormal = sess.run([crossEntropyError, accuracy, accuracyNormal], feed_dict={X: np.reshape(self.validData, (self.validData.shape[0],784)), Xall: np.reshape(self.trainData, (3500, 784)), y_targetAll: self.trainTarget, y_target: self.validTarget, needTrain: False})

                errTest, accTest, accTestNormal = sess.run([crossEntropyError, accuracy, accuracyNormal], feed_dict={X: np.reshape(self.testData, (self.testData.shape[0], 784)), Xall: np.reshape(self.trainData, (3500, 784)), y_targetAll: self.trainTarget, y_target: self.testTarget, needTrain: False})
                yValidErr.append(errValid)
                yTestErr.append(errTest)
                yValidAcc.append(accValid)
                yValidNormalAcc.append(accValidNormal)
                yTestAcc.append(accTest)
                yTestNormalAcc.append(accTestNormal)
            currEpoch += 1
        print "Binary-Class"
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
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yTrainErr))
        plt.savefig("BinaryTrainLossLearnRate" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')
        plt.figure(figureCount)
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yValidErr))
        plt.savefig("BinaryValidLossLearnRate" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')
        plt.figure(figureCount)
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yTestErr))
        plt.savefig("BinaryTestLossLearnRate" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')

        plt.figure(figureCount)
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yTrainAcc))
        plt.plot(np.array(xAxis), np.array(yTrainNormalAcc))
        plt.savefig("BinaryTrainAccuracy" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')
        plt.figure(figureCount)
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yValidAcc))
        plt.plot(np.array(xAxis), np.array(yValidNormalAcc))
        plt.savefig("BinaryValidAccuracy" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')
        plt.figure(figureCount)
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yTestAcc))
        plt.plot(np.array(xAxis), np.array(yTestNormalAcc))
        plt.savefig("BinaryTestAccuracy" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')
        return max(np.array(yTestAcc))

    def LogisticRegressionMethodMulti(self):
        """
        Implements logistic regression and cross-entropy loss.
        using tf.nn.sigmoid_cross_entropy_with_logits
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
        self.numEpoch = self.numEpochMulti
        W, b, X, y_target, y_predicted, crossEntropyError, train , needTrain, accuracy = self.buildGraphMulti()
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
                _, errTrain, currentW, currentb, yhat, accTrain= sess.run([train, crossEntropyError, W, b, y_predicted, accuracy], feed_dict={X: np.reshape(self.trainData[step*self.miniBatchSize:(step+1)*self.miniBatchSize], (self.miniBatchSize,784)),y_target: self.trainTarget[step*self.miniBatchSize:(step+1)*self.miniBatchSize], needTrain: True})
                wList.append(currentW)
                step = step + 1
                xAxis.append(numUpdate)
                numUpdate += 1
                yTrainErr.append(errTrain)
                yTrainAcc.append(accTrain)
                # These will not optimize the function cause you did not fetch 'train' 
                # So it won't have to execute that.
                errValid, accValid = sess.run([crossEntropyError, accuracy], feed_dict={X: np.reshape(self.validData, (self.validData.shape[0],784)), y_target: self.validTarget, needTrain: False})

                errTest, accTest = sess.run([crossEntropyError, accuracy], feed_dict={X: np.reshape(self.testData, (self.testData.shape[0], 784)), y_target: self.testTarget, needTrain: False})
                yValidErr.append(errValid)
                yTestErr.append(errTest)
                yValidAcc.append(accValid)
                yTestAcc.append(accTest)
            currEpoch += 1
        print "Multi-Class"
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
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yTrainErr))
        plt.savefig("MultiTrainLossLearnRate" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')
        plt.figure(figureCount)
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yValidErr))
        plt.savefig("MultiValidLossLearnRate" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')
        plt.figure(figureCount)
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yTestErr))
        plt.savefig("MultiTestLossLearnRate" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')

        plt.figure(figureCount)
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yTrainAcc))
        plt.savefig("MultiTrainAccuracy" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')
        plt.figure(figureCount)
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yValidAcc))
        plt.savefig("MultiValidAccuracy" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')
        plt.figure(figureCount)
        figureCount = figureCount + 1
        plt.plot(np.array(xAxis), np.array(yTestAcc))
        plt.savefig("MultiTestAccuracy" + str(self.learningRate) + "Batch" + str(self.miniBatchSize) + '.png')
        return max(np.array(yTestAcc))

    def buildGraphMulti(self):
        # Parameters to train
        # Images are 28*28 = 784 pixels

        # Multi-class Clasification (10 classes)
        # note: very important to increase dimensions W & b to train each 
        # separately using cross entropy error
        W = tf.Variable(tf.truncated_normal(shape=[784, 10], stddev=0.5), name='weights')
        b = tf.Variable(tf.zeros([10]), name='biases')

        # Supervised Inputs
        X = tf.placeholder(tf.float32, [None, 784], name='input_x')

        # y_target for binary class classification
        y_target = tf.placeholder(tf.float32, [None, 10], name='target_y')

        # Label to know if it should train or simply return the errors
        needTrain = tf.placeholder(tf.bool)

        weightDecayCoeff = tf.div(tf.constant(self.weightDecay),tf.constant(2.0))

        # Graph definition
        y_predicted = tf.matmul(X, W) + b
        
        # Multi-class Classification
        correctPred = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_target, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, "float"))

        # Weight Decay Error calculation
        weightDecayMeanSquareError = tf.reduce_mean(tf.square(W))
        weightDecayError = tf.multiply(weightDecayCoeff, weightDecayMeanSquareError)

        # Cross Entropy Softmax Error Multi-class Classification
        # note: Cross entropy only works with values from 0 to 1, so multi-class must be one hot encoded
        crossEntropySoftmaxError = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_predicted, y_target))
        crossEntropySoftmaxError = tf.add(crossEntropySoftmaxError, weightDecayError)

        # Don't train if don't have to for validation and test set
        #finalTrainingError = tf.select(needTrain, meanSquaredError, tf.constant(0.0))
        #finalTrainingError = tf.select(needTrain, crossEntropySigmoidError, tf.constant(0.0))
        finalTrainingError = tf.select(needTrain, crossEntropySoftmaxError, tf.constant(0.0))

        # Training mechanism
        #gdOptimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learningRate)
        adamOptimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate)
        
        # Train and update the parameters defined
        # sess.run(train) will execute the optimized function
        train = adamOptimizer.minimize(loss=finalTrainingError)

        return W, b, X, y_target, y_predicted, crossEntropySoftmaxError, train, needTrain, accuracy

    # Build the computational graph
    def buildGraphBinary(self):
        # Parameters to train
        # Images are 28*28 = 784 pixels

        # TODO: Update the W and b initialization so it depends on inputTarget size
        # Binary Classification
        W = tf.Variable(tf.truncated_normal(shape=[784, 1], stddev=0.5), name='weights')
        b = tf.Variable(0.0, name='biases')
        # Supervised Inputs
        X = tf.placeholder(tf.float32, [None, 784], name='input_x')
        # Need all for 1.1.3
        Xall = tf.placeholder(tf.float32, [None, 784], name='input_x')

        # y_target for binary class classification
        y_target = tf.placeholder(tf.float32, [None,1], name='target_y')
        # Need all for 1.1.3
        y_targetAll = tf.placeholder(tf.float32, [None,1], name='target_y')

        # Label to know if it should train or simply return the errors
        needTrain = tf.placeholder(tf.bool)

        weightDecayCoeff = tf.div(tf.constant(self.weightDecay),tf.constant(2.0))
        # Graph definition
        y_predicted = tf.matmul(X, W) + b

        # This works only for Binary Case
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

        # Cross Entropy Sigmoid Error Binary-class Calculation
        crossEntropySigmoidError = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_predicted, y_target))
        crossEntropySigmoidError = tf.add(crossEntropySigmoidError, weightDecayError)

        # Don't train if don't have to for validation and test set

        # For 1.1.3 to be updated to normal equation instead
        # This was for your own gradient descent for MSE that was not needed in this assignment
        #finalTrainingError = tf.select(needTrain, meanSquaredError, tf.constant(0.0))

        finalTrainingError = tf.select(needTrain, crossEntropySigmoidError, tf.constant(0.0))

        # Training mechanism

        # For 1.1.1
        # TODO: Plot by manually tweaking between both optimizers
        #gdOptimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learningRate)

        # For 1.1.2 onwards
        adamOptimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate)
        
        # Train and update the parameters defined
        # sess.run(train) will execute the optimized function
        # train = optimizer.minimize(loss=finalTrainingError)
        train = adamOptimizer.minimize(loss=finalTrainingError)

        # Return both errors for plotting for 1.1.3 and set weight to 0 for LogisticRegression for plotting when comparing
        # 1.1.3 Calculate Linear Regression using Normal Equation (analytical solution) 
        # Concatenate 1 to account for Bias
        OnesForX = tf.ones(shape=tf.pack([tf.shape(Xall)[0], 1]))
        Xnormal = tf.concat(1,[Xall , OnesForX])
        WNormalEquation = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(Xnormal), Xnormal)), tf.transpose(Xnormal)), y_targetAll)
        OnesForCurrX = tf.ones(shape=tf.pack([tf.shape(X)[0], 1]))
        currX = tf.concat(1,[X , OnesForCurrX])
        y_predictedNormal = tf.matmul(currX, WNormalEquation)

        correctPredNormal = tf.equal(tf.cast(tf.greater_equal(y_predictedNormal, 0.5), tf.float32), tf.floor(y_target))
        accuracyNormal = tf.reduce_mean(tf.cast(correctPredNormal, "float"))

        return W, b, X, y_target, y_predicted, crossEntropySigmoidError, train, needTrain, accuracy, WNormalEquation, accuracyNormal, Xall, y_targetAll

    def ShuffleBatches(self, trainData, trainTarget):
        # Gets the state as the current time
        rngState = np.random.get_state()
        np.random.shuffle(trainData)
        np.random.set_state(rngState)
        np.random.shuffle(trainTarget)
        return trainData, trainTarget

def convertOneHot(targetValues):
    numClasses = np.max(targetValues) + 1
    return np.eye(numClasses)[targetValues]

if __name__ == "__main__":
    print "LogisticRegression"
    print "Takes about 90 seconds"

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
        # Binary Class Linear Regression =  0.001
        # Binary Class Logistic Regression Sigmoid = 0.01
        # Cross entropy loss only works from 0 to 1! Not from 0 to 9 for multi-class
        for learningRate in [0.01]:
            tf.reset_default_graph()
            l = LogisticRegression(trainData, trainTarget, validData, validTarget, testData, testTarget, learningRate)
            maxTestAccuracy = l.LogisticRegressionMethodBinary()
            print "Max Test Accuracy is: ", maxTestAccuracy

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
        # Target values are from 0 to 9
        testData, testTarget = Data[16000:], Target[16000:]
        trainTarget = convertOneHot(trainTarget)
        validTarget = convertOneHot(validTarget)
        testTarget = convertOneHot(testTarget)
        # Multiclass Classification Logistic Regression Softmax = 0.01
        for learningRate in [0.01]:
            tf.reset_default_graph()
            l = LogisticRegression(trainData, trainTarget, validData, validTarget, testData, testTarget, learningRate)
            maxTestAccuracy = l.LogisticRegressionMethodMulti()
            print "Max Test Accuracy is: ", maxTestAccuracy
