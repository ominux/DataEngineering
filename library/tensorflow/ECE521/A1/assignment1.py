# Assignment 1 
# Optimization
# Early Stopping
# Learning rate decay
# Momentum

import tensorflow as tf
import numpy as np
import sys

# 1.2 Euclidean Distance Function 
# 1.2.2 Pairwise Distances
# Write a vectorized Tensorflow Python function that implements
# the pairwise squared Euclidean distance function for two input matrices.
# No Loops and makes use of Tensorflow broadcasting.
def PairwiseDistances(X, Z):
    """
    input:
        X is a matrix of size (B x N)
        Z is a matrix of size (C x N)
    output:
        D = matrix of size (B x C) containing the pairwise Euclidean distances
    """
    B = X.get_shape().as_list()[0]
    N = X.get_shape().as_list()[1]
    C = Z.get_shape().as_list()[0]
    # Ensure the N dimensions are consistent
    assert  N == Z.get_shape().as_list()[1]
    # Reshape to make use of broadcasting in Python
    X = tf.reshape(X, [B, 1, N])
    Z = tf.reshape(Z, [1, C, N])
    # The code below automatically does broadcasting
    D = tf.reduce_sum(tf.square(tf.subtract(X, Z)), 2)
    return D

# 1.3 Making Predictions
# 1.3.1 Choosing nearest neighbours
# Write a vectorized Tensorflow Python function that takes a pairwise distance matrix
# and returns the responsibilities of the training examples to a new test data point. 
# It should not contain loops.
# Use tf.nn.top_k
def ChooseNearestNeighbours(D, K):
    """
    input:
        D is a matrix of size (B x C)
        K is the top K responsibilities for each test input
    output:
        topK are the value of the squared distances for the topK
        indices are the index of the location of these squared distances
    """
    # Take topK of negative distances since it is the closest data.
    topK, indices = tf.nn.top_k(tf.negative(D), K)
    return topK, indices

# 1.3.2 Prediction
# Compute the k-NN prediction with K = {1, 3, 5, 50}
# For each value of K, compute and report:
    # training MSE loss
    # validation MSE loss
    # test MSE loss
# Choose best k using validation error = 50
def PredictKnn(trainData , testData, trainTarget,  testTarget, K):
    """
    input:
        trainData
        testData
        trainTarget
        testTarget
    output:
        loss
    """
    D = PairwiseDistances(testData, trainData)
    topK, indices = ChooseNearestNeighbours(D, K)
    # Select the proper outputs to be averaged from the target values and average them
    trainTargetSelectedAveraged = tf.reduce_mean(tf.gather(trainTarget, indices), 1)
    # Calculate the loss from the actual values
    # Divide by 2.0 since it's average over 2M instead of M where M = number of training data.
    loss = tf.reduce_mean(tf.square(tf.subtract(trainTargetSelectedAveraged, testTarget)))/2.0
    return loss

# Plot the prediction function for x = [0, 11] on training data.
def PredictedValues(x, trainData, trainTarget, K):
    """
    Plot the predicted values
    input:
        x = test target to plot and predict
    """
    D = PairwiseDistances(x, trainData)
    topK, indices = ChooseNearestNeighbours(D, K)
    predictedValues = tf.reduce_mean(tf.gather(trainTarget, indices), 1)
    return predictedValues

# 1.4 Soft-Knn & Gaussian Processes
# 1.4.1.1 Soft Decisions
# Write a Tensorflow python program based on the soft k-NN model to compute 
# predictions on the data1D.npy dataset. 
# Set lambda = 100 NOT 10 as given in assignment handout
# and plot the test-set prediction of the model. 

# Predict values using soft decision
def PredictedValuesSoftDecision(x, trainData, trainTarget):
    hyperParam = 100
    D1 = PairwiseDistances(x, trainData)
    K1 =  tf.exp(-hyperParam*D1)
    sum1 = tf.reduce_sum(tf.transpose(K1), axis=0)
    N = sum1.get_shape().as_list()[0]
    sum1 = tf.reshape(sum1, [N,1])
    rStar = tf.div(K1, sum1)
    predictedValues = tf.matmul(rStar,trainTarget)
    return predictedValues

# Predict values using Gaussian
# 1.4.1.1 Gaussian Processes
def PredictedValuesGaussianProcesses(x, trainData, trainTarget):
    hyperParam = 100
    D1 = PairwiseDistances(x, trainData)
    K1 =  tf.exp(-hyperParam*D1)
    D2 = PairwiseDistances(trainData, trainData)
    K2 =  tf.matrix_inverse(tf.exp(-hyperParam*D2))
    rStar = tf.matmul(K1, K2)
    predictedValues = tf.matmul(rStar,trainTarget)
    return predictedValues

# Comment on the difference you observe between two programs
# Gaussian has higher loss. 

# 2 Linear and Logistic Regression
# 2.2 Stochastic Gradient Descent
# Implement linear regression and stochastic gradient descent algorithm 
# with mini-batch size B = 50.
def buildGraph(learningRate, weightDecayCoeff):
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[64, 1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 64], name='input_x')
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')
    weightDecay = tf.div(tf.constant(weightDecayCoeff),tf.constant(2.0))
    # Graph definition
    y_predicted = tf.matmul(X,W) + b 
    # Error definition
    # Divided by 2M instead of M
    meanSquaredError = tf.div(tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target), 
                                                reduction_indices=1, 
                                                name='squared_error'), 
                                  name='mean_squared_error'), tf.constant(2.0))
    weightDecayMeanSquareError = tf.reduce_mean(tf.square(W))
    weightDecayTerm = tf.multiply(weightDecay, weightDecayMeanSquareError)
    meanSquaredError = tf.add(meanSquaredError,weightDecayTerm)

    # Training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learningRate)
    train = optimizer.minimize(loss=meanSquaredError)
    return W, b, X, y_target, y_predicted, meanSquaredError, train


def ShuffleBatches(trainData, trainTarget):
    rngState = np.random.get_state()
    np.random.shuffle(trainData)
    np.random.set_state(rngState)
    np.random.shuffle(trainTarget)
    return trainData, trainTarget

def LinearRegression(trainData, trainTarget, validData, validTarget, testData, testTarget):
    figureCount = 30
    # 2.2.3 Generalization (done by partner) 
    # Run SGD with B = 50 and use validation performance to choose best weight decay coefficient
    # from weightDecay = {0., 0.0001, 0.001, 0.01, 0.1, 1.}
    # Plot weightDecay vs test set accuracy. (Done by partner) 
    weightDecayTrials= [0.0, 0.0001, 0.0001, 0.01, 0.1, 1.0]
    # Plot total loss function vs number of updates for the best learning rate found
    learningRateTrials = [0.1, 0.01, 0.001]
    # 2.2.2 Effect of the mini-batch size
    # Run with Batch Size, B = {10, 5, 100, 700} and tune the learning rate separately for each mini-batch size.
    # Plot  the total loss function vs the number of updates for each mini-batch size.
    miniBatchSizeTrials = [10, 50, 100, 700]
    learningRate = 0.01
    miniBatchSize = 10
    weightDecayCoeff = 1.0
    # for weightDecayCoeff in weightDecayTrials:
    for miniBatchSize in miniBatchSizeTrials:
        for learningRate in learningRateTrials:
            # Reset entire graph
            tf.reset_default_graph()
            # Build computation graph
            W, b, X, y_target, y_predicted, meanSquaredError, train = buildGraph(learningRate, weightDecayCoeff)
            # Initialize session
            init = tf.global_variables_initializer()
            sess = tf.InteractiveSession()
            sess.run(init)
            initialW = sess.run(W)  
            initialb = sess.run(b)

            # print "Initial weights: %s, initial bias: %.2f", initialW, initialb
            # Training model
            numEpoch = 200
            currEpoch = 0
            wList = []

            xAxis = []
            yTrainErr = []
            yValidErr = []
            yTestErr = []
            numUpdate = 0
            step = 0
            errTrain = -1 
            errValid = -1 
            errTest = -1 
            while currEpoch <= numEpoch:
                # Shuffle the batches and return
                trainData, trainTarget = ShuffleBatches(trainData, trainTarget)
                step = 0 
                # Full batch
                while step*miniBatchSize < 700:
                    _, errTrain, currentW, currentb, yhat = sess.run([train, meanSquaredError, W, b, y_predicted], feed_dict={X: trainData[step*miniBatchSize:(step+1)*miniBatchSize], y_target: trainTarget[step*miniBatchSize:(step+1)*miniBatchSize]})
                    wList.append(currentW)
                    #if not (step*miniBatchSize % 50):
                    #    print "Iter: %3d, MSE-train: %4.2f, weights: %s, bias: %.2f", step, err, currentW.T, currentb
                    step = step + 1
                    xAxis.append(numUpdate)
                    numUpdate += 1
                    yTrainErr.append(errTrain)
                    # Note: These are not training the model since you did not fetch 'train'
                    errValid = sess.run(meanSquaredError, feed_dict={X: validData, y_target: validTarget})
                    errTest = sess.run(meanSquaredError, feed_dict={X: testData, y_target: testTarget})
                    yValidErr.append(errValid)
                    yTestErr.append(errTest)
                # Testing model
                # TO know what is being run
                currEpoch += 1
            print "LearningRate: " , learningRate, " Mini batch Size: ", miniBatchSize
            print "Iter: ", numUpdate
            print "Final Train MSE: ", errTrain
            print "Final Valid MSE: ", errValid
            print "Final Test MSE: ", errTest
            import matplotlib.pyplot as plt
            plt.figure(figureCount)
            figureCount = figureCount + 1
            plt.plot(np.array(xAxis), np.array(yTrainErr))
            plt.savefig("TrainLossLearnRate" + str(learningRate) + "Batch" + str(miniBatchSize) + '.png')

            plt.figure(figureCount)
            figureCount = figureCount + 1
            plt.plot(np.array(xAxis), np.array(yValidErr))
            plt.savefig("ValidLossLearnRate" + str(learningRate) + "Batch" + str(miniBatchSize) + '.png')
            plt.figure(figureCount)
            figureCount = figureCount + 1
            plt.plot(np.array(xAxis), np.array(yTestErr))
            plt.savefig("TestLossLearnRate" + str(learningRate) + "Batch" + str(miniBatchSize) + '.png')
    return

def SortData(inputVal, outputVal):
    """
    This sorts a given test set by the dataValue before plotting it.
    """
    p = np.argsort(inputVal, axis=0)
    inputVal = np.array(inputVal)[p]
    outputVal = np.array(outputVal)[p]
    inputVal = inputVal[:, :,0]
    outputVal = outputVal[:, :,0]
    return inputVal, outputVal

if __name__ == "__main__":
    print 'helloworld'
    N = 2 # number of dimensions
    B = 3 # number of test inputs (To get the predictions for all these inputs
    C = 2 # number of training inputs (Pick closest k from this C)
    X = tf.constant([1, 2, 3, 4, 5, 6], shape=[3, 2])
    Z = tf.constant([21, 22, 31, 32], shape=[2, 2])
    # Need to put seed so random_uniform doesn't generate new random values
    # each time you evaluate when you print, so then the values would be 
    # inconsistent as to what you would have used or checked
    #X = tf.random_uniform([B, N], seed=111)*30
    #Z = tf.random_uniform([C, N], seed=112)*30
    D = PairwiseDistances(X, Z)
    K = 1 # number of nearest neighbours
    # You calculate all the pairwise distances between each test input
    # and existing training input
    topK, indices = ChooseNearestNeighbours(D, K)
    # Prediction
    #for K in [1, 3, 5, 50]:
    for K in [1]:
        np.random.seed(521)
        Data = np.linspace(1.0 , 10.0 , num =100) [:, np.newaxis]
        Target = np.sin( Data ) + 0.1 * np.power( Data , 2) + 0.5 * np.random.randn(100 , 1)
        randIdx = np.arange(100)
        np.random.shuffle(randIdx)
        # data1D.npy
        trainData, trainTarget  = Data[randIdx[:5]], Target[randIdx[:5]]
        trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
        validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
        testData, testTarget = Data[randIdx[90:93]], Target[randIdx[90:93]]
        testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

        #trainData, trainTarget = SortData(trainData, trainTarget)
        #validData, validTarget = SortData(validData, validTarget)
        testData, testTarget = SortData(testData, testTarget)


        # Convert to tensors from numpy
        trainData = tf.stack(trainData)
        validData = tf.stack(validData)
        testData = tf.stack(testData)
        trainTarget = tf.stack(trainTarget)
        validtarget = tf.stack(validTarget)
        testTarget = tf.stack(testTarget)
        trainMseLoss = PredictKnn(trainData, trainData, trainTarget, trainTarget, K)
        validationMseLoss = PredictKnn(trainData, validData, trainTarget, validTarget, K)
        testMseLoss = PredictKnn(trainData, testData, trainTarget, testTarget, K)
        init = tf.global_variables_initializer()
        '''
        with tf.Session() as sess:
            sess.run(init)
            print 'K ' + str(K)
            print 'trainMseLoss'
            print sess.run(trainMseLoss)
            print 'validationMseLoss'
            print sess.run(validationMseLoss)
            print 'testMseLoss'
            print sess.run(testMseLoss)
        '''
        # Plot the prediction for the x below
        x = np.linspace(0.0, 11.0, num=1000)[:, np.newaxis]
        xTensor = tf.stack(x)
        predictedValuesKnn = PredictedValues(xTensor, trainData, trainTarget, K)
        predictedValuesSoft = PredictedValuesSoftDecision(testData, trainData, trainTarget)
        predictedValuesGaussian = PredictedValuesGaussianProcesses(testData, trainData, trainTarget)
        lossSoft = tf.reduce_mean(tf.square(tf.subtract(predictedValuesSoft, testTarget)))/2.0
        lossGaussian = tf.reduce_mean(tf.square(tf.subtract(predictedValuesGaussian, testTarget)))/2.0
        import matplotlib.pyplot as plt
        plt.figure(0)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            plt.figure(K+100)
            plt.scatter(sess.run(trainData), sess.run(trainTarget))
            plt.plot(sess.run(xTensor), sess.run(predictedValuesKnn))
            fileName = str("KNN") + str(K) + str("trainingGraph.png")
            plt.savefig(fileName)

            # Plot for SoftDecision
            plt.figure(K+101)
            plt.title("Soft Decision kNN on Test Set, MSE = " + str(sess.run(lossSoft)))
            plt.xlabel("Data Value")
            plt.ylabel("Target Value")
            plt.scatter(sess.run(testData), sess.run(testTarget), label= "testValue")
            plt.plot(sess.run(testData), sess.run(predictedValuesSoft), label = "predicted")
            plt.legend()
            fileName = str("SoftDecision.png")
            plt.savefig(fileName)
            print 'SoftDecisionLoss'
            print sess.run(lossSoft)

            # Plot for Gaussian
            plt.figure(K+102)
            plt.title("Gaussian Process Regression on Test Set, MSE = " + str(sess.run(lossGaussian)))
            plt.xlabel("Data Value")
            plt.ylabel("Target Value")
            plt.scatter(sess.run(testData), sess.run(testTarget), label = "testValue")
            plt.plot(sess.run(testData), sess.run(predictedValuesGaussian), label = "predicted")
            plt.legend()
            fileName = str("ConditionalGaussian.png")
            plt.savefig(fileName)
            print 'ConditionalGaussianLoss'
            print sess.run(lossGaussian)
    # Part 2
    with np.load ("tinymnist.npz") as data :
        trainData, trainTarget = data ["x"], data["y"]
        validData, validTarget = data ["x_valid"], data ["y_valid"]
        testData, testTarget = data ["x_test"], data ["y_test"]
        LinearRegression(trainData, trainTarget,validData, validTarget, testData, testTarget)
