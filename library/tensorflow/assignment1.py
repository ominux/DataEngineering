# Assignment 1 
# Optimization
# Early Stopping
# Learning rate decay
# Momentum

import tensorflow as tf
import numpy as np

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
    assert  N == Z.get_shape().as_list()[1]
    X = tf.reshape(X, [B, 1, N])
    Z = tf.reshape(Z, [1, C, N])
    D = tf.reduce_sum(tf.square(tf.sub(X, Z)), 2)
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
    topK, indices = tf.nn.top_k(D, K)
    return topK, indices

# 1.3.2 Prediction
# Compute the k-NN prediction with K = {1, 3, 5, 50}
# For each value of K, compute and report:
    # training MSE loss
    # validation MSE loss
    # test MSE loss
# Choose best k using validation error = 50
def PredictKnn(inputData, testData, inputTarget,  testTarget, K):
    """
    input:
        inputData
        testData
        inputTarget
        testTarget
    output:
        loss
    """
    D = PairwiseDistances(testData, trainData)
    topK, indices = ChooseNearestNeighbours(D, K)
    # Select the proper outputs to be averaged from the target values and average them
    trainTargetSelectedAveraged = tf.reduce_mean(tf.gather(trainTarget, indices), 1)
    # Calculate the loss from the actual values
    loss = tf.reduce_mean(tf.square(tf.sub(trainTargetSelectedAveraged, testTarget)))
    return loss

# TODO: Plot the prediction function for x = [0, 11]
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
    K = 5 # number of nearest neighbours
    np.random.seed(521)
    Data = np.linspace(1.0 , 10.0 , num =100) [:, np.newaxis]
    Target = np.sin( Data ) + 0.1 * np.power( Data , 2) + 0.5 * np.random.randn(100 , 1)
    randIdx = np.arange(100)
    np.random.shuffle(randIdx)
    trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
    validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
    testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]
    # Convert to tensors from numpy
    trainData = tf.pack(trainData)
    validData = tf.pack(validData)
    testData = tf.pack(testData)
    trainTarget = tf.pack(trainTarget)
    validtarget = tf.pack(validTarget)
    testTarget = tf.pack(testTarget)
    trainMseLoss = PredictKnn(trainData, trainData, trainTarget, trainTarget, K)
    validationMseLoss = PredictKnn(trainData, validData, trainTarget, validTarget, K)
    testMseLoss = PredictKnn(trainData, testData, trainTarget, testTarget, K)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print sess.run(trainMseLoss)
        print sess.run(validationMseLoss)
        print sess.run(testMseLoss)
    # Plot the prediction for the x below
    x = np.linspace(0.0, 11.0, num=1000)[:, np.newaxis]
    xTensor = tf.pack(x)
    predictedValues = PredictedValues(xTensor, trainData, trainTarget, K)
    import matplotlib.pyplot as plt
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        predicted = sess.run(predictedValues)
        plt.plot(x, predicted)
        plt.scatter(sess.run(trainData), sess.run(trainTarget))
        plt.savefig('haha.png')

    '''
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        opX = sess.run(X)
        opZ = sess.run(Z)
        opD = sess.run(D)
        print 'X'
        print X
        print 'Z'
        print Z
        print 'D'
        print D
        print 'X'
        print opX
        print 'Z'
        print opZ
        print 'D'
        print opD
        sess.run(init)
        opD = sess.run(D)
        opTopK = sess.run(topK)
        opIndices = sess.run(indices)
        print D
        print opD
        print topK
        print opTopK
        print indices
        print opIndices
        # Calculating loss
        sess.run(init)
        opTopK = sess.run(topK)
        opIndices = sess.run(indices)
        opTrainTar = sess.run(trainTargetSelectedAveraged)
        opLoss = sess.run(loss)
        print 'indices are'
        print opIndices
        print 'trainTarSelectedAveraged are'
        print opTrainTar
        print 'Loss is'
        print opLoss
        # Don't use topK as it's just the calculated distances
        # you need to use opIndices to index into the numpyMatrices
        print 'topK are'
        print opTopK
        # Get all the values for the nearest k target and sum them up
        # trainTarget[indices]
    '''


'''

# 1.4 Soft-Knn & Gaussian Processes
# 1.4.1.1 Soft Decisions
# TODO: Write a Tensorflow python program based on the soft k-NN model to compute 
# predictions on the data1D.npy dataset. 
# Set lambda = 1 and plot the test-set prediction of the model. 
# 1.4.1.1 Gaussian Processes
# TODO: Repeat for the Gaussian Processes Regression Model
# Comment on the difference you observe between two programs


# 2 Linear and Logistic Regression
# 2.2 Stochastic Gradient Descent

# TODO: Implement linear regression and stochastic gradient descent algorithm 
# with mini-batch size B = 50.

with np.load ("tinymnist.npz") as data :
    trainData, trainTarget = data ["x"], data["y"]
    validData, validTarget = data ["x_valid"], data ["y_valid"]
    testData, testTarget = data ["x_test"], data ["y_test"]

# 2.2.1 Tuning the learning rate
# Train the linear regression model on the tiny MNIST dataset using SGD
# to optimize for the total loss.
# Set weight decay = 1
# Tune learning rate to obtain best overall convergence speed.
# TODO: Plot total loss function vs number of updates for the best learning rate found

# 2.2.2 Effect of the mini-batch size
# TODO: Run with Batch Size, B = {10, 5, 100, 700} and tune the learning rate separately for each mini-batch size.
# TODO: Plot  the total loss function vs the number of updates for each mini-batch size.

# 2.2.3 Generalization
# TODO: Run SGD with B = 50 and use validation performance to choose best weight decay coefficient
# from weightDecay = {0., 0.0001, 0.001, 0.01, 0.1, 1.}

# TODO: Plot weightDecay vs test set accuracy. 
'''
