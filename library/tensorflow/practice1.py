# Book: Hello-Tensorflow Oreilly

import tensorflow as tf

# The Default Graph has been initialized when you imported tensorflow
graph = tf.get_default_graph()
print graph # point to a memory address

# Nodes of the graphs are the operations
operations = graph.get_operations() # points to an empty graph, gets a copy, NOT reference
print operations

# Put a constant node of value 1
inputX = tf.constant(1.0)

# Every node has (type, dimensionsShape, dataType) 
print inputX  # A constant 32-bit float tensor of no dimension
# To evaluate the actual value of inputValue, need to use a session
session = tf.Session() # Returns the Default Graph
print session.run(inputX) # A session evaluates a given computational graph

print operations # It is a copy of the operations  that was empty
operations = graph.get_operations() # points to an empty graph
print operations # now it points to a memory address of the first node
print "Printing first node's definition"
print operations[0].node_def # Print's it in Google's Protocol Buffer format

# Weights or Parameters are called Variables in tensorflow
weight = tf.Variable(0.8)

print "All Operations"
for currOperation in graph.get_operations(): 
    # Variables adds 4 operations in the graph    
    print currOperation.name

# Multiply them together
predictedOutputY = tf.mul(weight, inputX)

print "Latest Operation and Inputs"
operations = graph.get_operations()[-1] # Gets lastest operation`
print operations.name
for inputsForLastOperation in operations.inputs:
    print inputsForLastOperation

print "Learning"
# To train it to learn, define the supervised labeled output
correctOutputY = tf.constant(0.0)
squaredLossFunction = (predictedOutputY - correctOutputY)**2

# Optimize using gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.025)

# It calculates the derivative of the loss function.
# Then, it calculates the gradient (derivative of loss function)
# via back propagation 
gradientAndVariable = optimizer.compute_gradients(squaredLossFunction)

print gradientAndVariable

session.run(tf.initialize_all_variables())

session.run(gradientAndVariable[1][0])
print session

# Apply the gradients to the entire network.
session.run(optimizer.apply_gradients(gradientAndVariable))

# Check the new values of the weights after learning a single iteration
session.run(weight)


# It's hard to keep track of your computational graph
# Therefore, can use TensorBoard, which is a visualization tool.
print "Visualizing Outputs"
# Creates a directory called logSimpleGraph to store the current graph
tf.train.SummaryWriter('logSimpleGraph', session.graph)
print "To visualize Tensorboard, execute on commandline:"
# It will execute the visualization tool as a webserver for a browser
print ">> tensorboard --logdir=logSimpleGraph"













