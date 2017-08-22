# TODO: Implement this
class LogisticRegression(object):
  """Multi-class Logistic Regression Class
    Classification is done by projecting data points onto a set of hyperplanes, the distance is used to determine
    a class membership probability. 
  """
  def __init__(self, x, n_in, n_out):
    # Weights
    self.W = 2
    # Bias
    self.B = 1

  def getWeight(self):
    return self.W

  def setWeight(self, W):
    self.W = W

  def getBias(self):
    return self.B

  def setBias(self, B):
    self.B = B

  def __str__(self):
    return "Weight: %s Bias: %s" (self.W, self.B)
