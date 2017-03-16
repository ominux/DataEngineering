import datetime
import numpy as np
import tensorflow as tf
import sys

# TODO: Replace all 784 with some automatic calculation for pixel size
class LogisticRegression(object):
    def __init__(self, trainData, trainTarget, validData, validTarget, testData, testTarget, numEpoch = 1, learningRate = 0.0001, weightDecay = 0.01, optimizerType = "adam", classifierType = "Binary", executeLinearRegression = False, questionTitle = ""):
        self.trainData = trainData
        self.trainTarget = trainTarget
        self.validData = validData
        self.validTarget = validTarget
        self.testData = testData
        self.testTarget = testTarget
        self.miniBatchSize = 500
        self.numEpoch = numEpoch
        self.learningRate = learningRate
        self.weightDecay = weightDecay

        # To not execute Linear Regression every time
        self.executeLinearRegression = executeLinearRegression
        self.executedLinear = False
        self.WNormalEquationSave = 0
        self.accuracyNormalSave = 0
    
        self.questionTitle = questionTitle
        self.classifierType = classifierType
        self.optimizer = 0
        if optimizerType == "gd":
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learningRate)

    def printPlotResults(self, numUpdate, errTrain, accTrain, errValid, errTest, accValid, accTest, xAxis, yTrainErr, yValidErr, yTestErr, yTrainAcc, yValidAcc, yTestAcc, yTrainNormalAcc = 0, yValidNormalAcc = 0, yTestNormalAcc = 0):
        global figureCount
        import matplotlib.pyplot as plt
        print self.classifierType
        print "LearningRate: " , self.learningRate, " Mini batch Size: ", self.miniBatchSize
        print "NumEpoch: ", self.numEpoch
        print "Iter: ", numUpdate
        print "Final Train MSE: ", errTrain
        print "Final Train Acc: ", accTrain
        print "Final Valid MSE: ", errValid
        print "Final Test MSE: ", errTest
        print "Final Valid Acc: ", accValid
        print "Final Test Acc: ", accTest
        paramStr = "LearnRate"  + str(self.learningRate) + "NumEpoch" + str(self.numEpoch)
        typeLossStr = "Loss"
        typeAccuracyStr = "Accuracy"
        iterationStr = "Iteration"
        trainStr = "Train"
        validStr = "Valid"
        testStr = "Test"
        trainLossStr = trainStr + typeLossStr
        validLossStr = validStr + typeLossStr
        testLossStr = testStr + typeLossStr

        trainAccStr = trainStr + typeAccuracyStr
        validAccStr = validStr + typeAccuracyStr
        testAccStr = testStr + typeAccuracyStr

        linearLabelStr = "LinearRegression"

        figureCount = figureCount + 1
        plt.figure(figureCount)
        title = self.classifierType + typeLossStr + trainStr + paramStr
        plt.title(title)
        plt.xlabel(iterationStr)
        plt.ylabel(typeLossStr)
        plt.plot(np.array(xAxis), np.array(yTrainErr), label = trainLossStr)
        plt.legend()
        plt.savefig(self.questionTitle + title + ".png")
        plt.close()
        plt.clf()

        figureCount = figureCount + 1
        plt.figure(figureCount)
        title = self.classifierType + typeLossStr + validStr + paramStr
        plt.title(title)
        plt.xlabel(iterationStr)
        plt.ylabel(typeLossStr)
        plt.plot(np.array(xAxis), np.array(yValidErr), label = validLossStr)
        plt.legend()
        plt.savefig(self.questionTitle + title + ".png")
        plt.close()
        plt.clf()

        figureCount = figureCount + 1
        plt.figure(figureCount)
        title = self.classifierType + typeLossStr + testStr + paramStr
        plt.title(title)
        plt.xlabel(iterationStr)
        plt.ylabel(typeLossStr)
        plt.plot(np.array(xAxis), np.array(yTestErr), label = testLossStr)
        plt.legend()
        plt.savefig(self.questionTitle + title + ".png")
        plt.close()
        plt.clf()


        figureCount = figureCount + 1
        plt.figure(figureCount)
        title = self.classifierType + typeLossStr + trainStr + validStr + testStr + paramStr
        plt.title(title)
        plt.xlabel(iterationStr)
        plt.ylabel(typeLossStr)
        plt.plot(np.array(xAxis), np.array(yTrainErr), label = trainLossStr)
        plt.plot(np.array(xAxis), np.array(yValidErr), label = validLossStr)
        plt.plot(np.array(xAxis), np.array(yTestErr), label = testLossStr)
        plt.legend()
        plt.savefig(self.questionTitle + title + ".png")
        plt.close()
        plt.clf()

        # Accuracies
        figureCount = figureCount + 1
        plt.figure(figureCount)
        title = self.classifierType + typeAccuracyStr + trainStr + paramStr
        plt.title(title)
        plt.xlabel(iterationStr)
        plt.ylabel(typeAccuracyStr)
        plt.plot(np.array(xAxis), np.array(yTrainAcc), label = trainAccStr)
        accLabel = trainStr + typeAccuracyStr
        if self.executeLinearRegression:
            plt.plot(np.array(xAxis), np.array(yTrainNormalAcc), label = linearLabelStr)
        plt.legend()
        plt.savefig(self.questionTitle + title + ".png")
        plt.close()
        plt.clf()

        figureCount = figureCount + 1
        plt.figure(figureCount)
        title = self.classifierType + typeAccuracyStr + validStr + paramStr
        plt.title(title)
        plt.xlabel(iterationStr)
        plt.ylabel(typeAccuracyStr)
        plt.plot(np.array(xAxis), np.array(yValidAcc), label = validAccStr)
        if self.executeLinearRegression:
            plt.plot(np.array(xAxis), np.array(yValidNormalAcc), label = linearLabelStr)
        plt.legend()
        plt.savefig(self.questionTitle + title + ".png")
        plt.close()
        plt.clf()

        figureCount = figureCount + 1
        title = self.classifierType + typeAccuracyStr + testStr + paramStr
        plt.figure(figureCount)
        plt.title(title)
        plt.xlabel(iterationStr)
        plt.ylabel(typeAccuracyStr)
        plt.plot(np.array(xAxis), np.array(yTestAcc), label = testAccStr)
        if self.executeLinearRegression:
            plt.plot(np.array(xAxis), np.array(yTestNormalAcc), label = linearLabelStr)
        plt.legend()
        plt.savefig(self.questionTitle + title + ".png")
        plt.close()
        plt.clf()


        figureCount = figureCount + 1
        title = self.classifierType + typeAccuracyStr + trainStr + validStr + testStr + paramStr
        plt.figure(figureCount)
        plt.title(title)
        plt.xlabel(iterationStr)
        plt.ylabel(typeAccuracyStr)
        plt.plot(np.array(xAxis), np.array(yTrainAcc), label = trainAccStr)
        plt.plot(np.array(xAxis), np.array(yValidAcc), label = validAccStr)
        plt.plot(np.array(xAxis), np.array(yTestAcc), label = testAccStr)
        plt.legend()
        plt.savefig(self.questionTitle + title + ".png")
        plt.close()
        plt.clf()

        print self.questionTitle + self.classifierType
        print "Max Test Accuracy is: ", max(np.array(yTestAcc))

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
        W, b, X, y_target, y_predicted, crossEntropyError, train , needTrain, accuracy, WNormalEquation, accuracyNormal, Xall, y_targetAll = self.buildGraphBinary()

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
        while currEpoch < self.numEpoch:
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
            logStdOut("e" + str(currEpoch))
        self.printPlotResults(numUpdate, errTrain, accTrain, errValid, errTest, accValid, accTest, xAxis, yTrainErr, yValidErr, yTestErr, yTrainAcc, yValidAcc, yTestAcc, yTrainNormalAcc, yValidNormalAcc, yTestNormalAcc)

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
        W, b, X, y_target, y_predicted, crossEntropyError, train , needTrain, accuracy = self.buildGraphMulti()

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
        while currEpoch < self.numEpoch:
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
            logStdOut("e" + str(currEpoch))
        self.printPlotResults(numUpdate, errTrain, accTrain, errValid, errTest, accValid, accTest, xAxis, yTrainErr, yValidErr, yTestErr, yTrainAcc, yValidAcc, yTestAcc)
    
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
        crossEntropySoftmaxError = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_predicted, labels = y_target))
        crossEntropySoftmaxError = tf.add(crossEntropySoftmaxError, weightDecayError)

        # Don't train if don't have to for validation and test set
        #finalTrainingError = tf.select(needTrain, meanSquaredError, tf.constant(0.0))
        #finalTrainingError = tf.select(needTrain, crossEntropySigmoidError, tf.constant(0.0))
        finalTrainingError = tf.where(needTrain, crossEntropySoftmaxError, tf.constant(0.0))

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

        # TODO: Update the W and b initialization so it depends on inputTarget size instead of 784
        # Binary Classification
        W = tf.Variable(tf.truncated_normal(shape=[784, 1], stddev=0.5), name='weights')
        b = tf.Variable(0.0, name='biases')
        # Supervised Inputs
        X = tf.placeholder(tf.float32, [None, 784], name='input_x')

        # y_target for binary class classification
        y_target = tf.placeholder(tf.float32, [None,1], name='target_y')

        # Need all for 1.1.3
        Xall = tf.placeholder(tf.float32, [None, 784], name='input_x')
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

        '''
        # Mean Square Error Calculation
        # Divide by 2M instead of M
        meanSquaredError = tf.div(tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target), 
                                                    reduction_indices=1, 
                                                    name='squared_error'), 
                                      name='mean_squared_error'), tf.constant(2.0))
        meanSquaredError = tf.add(meanSquaredError, weightDecayError)
        '''

        # Cross Entropy Sigmoid Error Binary-class Calculation
        crossEntropySigmoidError = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_predicted, labels = y_target))
        crossEntropySigmoidError = tf.add(crossEntropySigmoidError, weightDecayError)

        # Don't train if don't have to for validation and test set

        # For 1.1.3 to be updated to normal equation instead
        # This was for your own gradient descent for MSE that was not needed in this assignment
        #finalTrainingError = tf.where(needTrain, meanSquaredError, tf.constant(0.0))

        finalTrainingError = tf.where(needTrain, crossEntropySigmoidError, tf.constant(0.0))

        # Training mechanism
        
        # Train and update the parameters defined
        # sess.run(train) will execute the optimized function
        train = self.optimizer.minimize(loss=finalTrainingError)

        # Return both errors for plotting for 1.1.3 and set weight to 0 for LogisticRegression for plotting when comparing
        WNormalEquation = W
        accuracyNormal = accuracy
        if self.executeLinearRegression:
            # 1.1.3 Calculate Linear Regression using Normal Equation (analytical solution) 
            # Concatenate 1 to account for Bias
            if self.executedLinear:
                WNormalEquation = self.WNormalEquationSave
                accuracyNormal = self.accuracyNormalSave

            else: 
                OnesForX = tf.ones(shape=tf.pack([tf.shape(Xall)[0], 1]))
                Xnormal = tf.concat(1,[Xall , OnesForX])
                WNormalEquation = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(Xnormal), Xnormal)), tf.transpose(Xnormal)), y_targetAll)
                OnesForCurrX = tf.ones(shape=tf.pack([tf.shape(X)[0], 1]))
                currX = tf.concat(1,[X , OnesForCurrX])
                y_predictedNormal = tf.matmul(currX, WNormalEquation)
                correctPredNormal = tf.equal(tf.cast(tf.greater_equal(y_predictedNormal, 0.5), tf.float32), tf.floor(y_target))
                accuracyNormal = tf.reduce_mean(tf.cast(correctPredNormal, "float"))
                self.WNormalEquationSave = WNormalEquation
                self.accuracyNormalSave = accuracyNormal
                self.executedLinear = True

        return W, b, X, y_target, y_predicted, crossEntropySigmoidError, train, needTrain, accuracy, WNormalEquation, accuracyNormal, Xall, y_targetAll

    def ShuffleBatches(self, trainData, trainTarget):
        # Gets the state as the current time
        rngState = np.random.get_state()
        np.random.shuffle(trainData)
        np.random.set_state(rngState)
        np.random.shuffle(trainTarget)
        return trainData, trainTarget

#---------------------------------------------------------------------------------------------
def convertOneHot(targetValues):
    numClasses = np.max(targetValues) + 1
    return np.eye(numClasses)[targetValues]

def ExecuteBinary(questionTitle, numEpoch, learningRates, weightDecay, optimizerType, executeLinearRegression):
    classifierType = "Binary"
    print questionTitle + classifierType
    for learningRate in learningRates:
        with np.load("notMNIST.npz") as data :
            # Need to get new data from scratch everything for a new learning rate
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
            tf.reset_default_graph()
            l = LogisticRegression(trainData, trainTarget, validData, validTarget, testData, testTarget, numEpoch, learningRate, weightDecay, optimizerType, classifierType, executeLinearRegression, questionTitle)
            l.LogisticRegressionMethodBinary()
            logElapsedTime(questionTitle  + classifierType + str(learningRate))

def ExecuteMulti(questionTitle, numEpoch, learningRates, weightDecay, optimizerType, executeLinearRegression):
    classifierType = "Multi"
    print questionTitle + classifierType
    with np.load("notMNIST.npz") as data:
        Data, Target = data ["images"], data["labels"]
        for learningRate in learningRates:
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
            tf.reset_default_graph()
            l = LogisticRegression(trainData, trainTarget, validData, validTarget, testData, testTarget, numEpoch, learningRate, weightDecay, optimizerType, classifierType, executeLinearRegression, questionTitle)
            l.LogisticRegressionMethodMulti()
            logElapsedTime(questionTitle  + classifierType + str(learningRate))

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
    logStdOut("LogisticRegression")
    # 0.0 LinearRegressionTraining
    # MeanSquareError learningRate = 0.001, otherwise overshoots 

    # 1 Logistic Regression
    # 1.1 Binary Classification, only 2 labels
    # CrossEntropyError, learningRate = 0.01, 98.6% test accuracy highest
    # Default Values
    '''
    numEpoch = 1
    optimizerType = "adam"
    executeLinearRegression = False
    '''

    '''
    questionTitle = "1.1.1" # Learning with GD
    optimizerType = "gd"
    numEpoch = 201
    learningRates = [0.1, 0.01, 0.001, 0.0001]
    weightDecay = 0.01
    executeLinearRegression = False
    logStdOut("Starting" + questionTitle)
    sys.stdout = open("result" + questionTitle + ".txt", "w") # write a new file from scratch
    ExecuteBinary(questionTitle, numEpoch, learningRates, weightDecay, optimizerType, executeLinearRegression)
    logStdOut("Finished" + questionTitle)
    # '''

    '''
    questionTitle = "1.1.2" # Beyond plain SGD
    optimizerType = "adam"
    numEpoch = 200
    learningRates = [0.1, 0.01, 0.001, 0.0001]
    weightDecay = 0.01
    executeLinearRegression = False
    logStdOut("Starting" + questionTitle)
    sys.stdout = open("result" + questionTitle + ".txt", "w")
    ExecuteBinary(questionTitle, numEpoch, learningRates, weightDecay, optimizerType, executeLinearRegression)
    logStdOut("Finished" + questionTitle)
    # '''

    '''
    # 1.1.3 
    questionTitle = "1.1.3" # Comparison With Linear Regression
    weightDecay = 0.00
    numEpoch = 200
    optimizerType = "adam"
    learningRates = [0.01]
    executeLinearRegression = True
    logStdOut("Starting" + questionTitle)
    sys.stdout = open("result" + questionTitle + ".txt", "w")
    ExecuteBinary(questionTitle, numEpoch, learningRates, weightDecay, optimizerType, executeLinearRegression)
    logStdOut("Finished" + questionTitle)
    # '''

    '''
    # Multi-class Classification, 10 labels
    # CrossEntropySoftmax Error, learningRate = 0.001
    questionTitle = "1.2.3" # Logistic Regression for multiclass
    weightDecay = 0.01
    numEpoch = 200 # Multi-class Classification
    optimizerType = "adam"
    learningRates = [1.0, 0.1, 0.01, 0.001, 0.0001]
    executeLinearRegression = False
    logStdOut("Starting" + questionTitle)
    sys.stdout = open("result" + questionTitle + ".txt", "w")
    ExecuteMulti(questionTitle, numEpoch, learningRates, weightDecay, optimizerType, executeLinearRegression)
    logStdOut("Finished" + questionTitle)
    # '''
