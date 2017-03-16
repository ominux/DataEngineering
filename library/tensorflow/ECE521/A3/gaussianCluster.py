import tensorflow as tf
import numpy as np
from dataInitializer import DataInitializer

class GaussianCluster(object):
    def __init__(self, K, trainData, validData, hasValid):
        """
        Constructor
        """
        self.K = K
        self.trainData = trainData
        self.validData = validData
        self.hasValid = hasValid

def reduce_logsumexp(input_tensor, reduction_indices=1, keep_dims=False):
  """Computes the sum of elements across dimensions of a tensor in log domain.
     
     It uses a similar API to tf.reduce_sum.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. 
    keep_dims: If true, retains reduced dimensions with length 1.
  Returns:
    The reduced tensor.
  """
  max_input_tensor1 = tf.reduce_max(input_tensor, 
                                    reduction_indices, keep_dims=keep_dims)
  max_input_tensor2 = max_input_tensor1
  if not keep_dims:
    max_input_tensor2 = tf.expand_dims(max_input_tensor2, 
                                       reduction_indices) 
  return tf.log(tf.reduce_sum(tf.exp(input_tensor - max_input_tensor2), 
                                reduction_indices, keep_dims=keep_dims)) + max_input_tensor1

if __name__ == "__main__":
    print "ECE521 Assignment 3: Unsupervised Learning: GaussianCluster"
    '''
    # Gaussian Cluster Model
    questionTitle = "2.1.2"
    # TODO: FIGURE THIS OUT, USES PART ONE STUFF
    # '''
    
    '''
    questionTitle = "2.1.3"
    diffK = [1 2 3 4 5]
    dataType = "2D"
    hasValid = True
    for K in diffK:
        executePartGaussianCluster(questionTitle, K, dataType, hasValid)
    # '''
