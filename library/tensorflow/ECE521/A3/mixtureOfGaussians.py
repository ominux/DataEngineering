import tensorflow as tf
import numpy as np
from dataInitializer import DataInitializer

class MixtureOfGaussians(object):
    def __init__(self, K, trainData, validData, hasValid):
        """
        Constructor
        """
        self.K = K
        self.trainData = trainData
        self.validData = validData
        self.hasValid = hasValid

def logsoftmax(input_tensor):
  """Computes normal softmax nonlinearity in log domain.

     It can be used to normalize log probability.
     The softmax is always computed along the second dimension of the input Tensor.     
 
  Args:
    input_tensor: Unnormalized log probability.
  Returns:
    normalized log probability.
  """
  return input_tensor - reduce_logsumexp(input_tensor, keep_dims=True)


if __name__ == "__main__":
    print "ECE521 Assignment 3: Unsupervised Learning: Mixture of Gaussian"
    '''
    # TODO:
    questionTitle = "2.2.2"
    # TODO:
    questionTitle = "2.2.3"
    # TODO:
    questionTitle = "2.2.4"
    # TODO:
    # '''
