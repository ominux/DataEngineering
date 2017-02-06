# Assignment 1 
# Optimization
# Early Stopping
# Learning rate decay
# Momentum

import tensorflow as tf
import numpy as np

# 1.2 Euclidean Distance Function 
# 1.2.2 Pairwise Distances
# TODO: Write a vectorized Tensorflow Python function that implements
# the pairwise squared Euclidean distance function for two input matrices.
# No Loops and makes used of Tensorflow broadcasting.

# 1.3 Making Predictions
# 1.3.1 Choosing nearest neighbours
# TODO: Write a vectorized Tensorflow Python function that takes a pairwise distance matrix
# and returns the responsibilities of the training examples to a new test data point. 
# It should not contain loops.
# Use tf.nn.top_k

# 1.3.2 Prediction
# TODO: Compute the k-NN prediction with K = {1, 3, 5, 50}
# For each value of K, compute and report:
    # training MSE loss
    # validation MSE loss
    # test MSE loss
# Choose best k using validation error
# TODO: Plot the prediction function for x = [0, 11]
x = np.linspace(0,0, 11.0, num=1000)[:, np.newaxis]

np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) + 0.5 * np.random.randn(100 , 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

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
