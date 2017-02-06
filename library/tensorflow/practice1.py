import tensorflow as tf

# The Default Graph has been initialized when you imported tensorflow
graph = tf.get_default_graph()
print graph # point to a memory address

# Tf.Tensors = Data that flows between oeprations
# TF.Operations = Nodes of the graphs 
operations = graph.get_operations() # points to an empty graph, gets a copy, NOT reference
print operations

# All operations works on the default graph until it is overriden
ownGraph = tf.Graph() # define a new graph
with ownGraph.as_default(): # All operations now work on ownGraph temporarily
    tf.constant(30.0)
ownGraph.finalize() # Finalize the state of the graph, making it read-only

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
lastOperation = graph.get_operations()[-1] # Gets lastest operation`
print lastOperation.name
for inputsForLastOperation in lastOperation.inputs:
    print inputsForLastOperation

print "Learning"
# To train it to learn, define the supervised labeled output
correctOutputY = tf.constant(0.0)
squaredLossFunction = tf.pow(predictedOutputY - correctOutputY, 2)

# Optimize using gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.025)

# It calculates the derivative of the loss function.
# Then, it calculates the gradient (derivative of loss function)
# via back propagation 
gradientAndVariable = optimizer.compute_gradients(squaredLossFunction)

print gradientAndVariable # Points to a memory address

session.run(tf.initialize_all_variables())
print session.run(gradientAndVariable)

# Outputs (gradientForWeight, weightValue)
print session.run(gradientAndVariable[0])

# Apply the gradients to the entire network.
session.run(optimizer.apply_gradients(gradientAndVariable))

# Check the new values of the weights after learning a single iteration
print session.run(weight)

# Can define a function to compute gradient for the loss function using minimize()
trainStepFunction = tf.train.GradientDescentOptimizer(0.025).minimize(squaredLossFunction)
# Keep applying the same function
for eachTrainStep in range(10):
    session.run(trainStepFunction)
    # Should approach 0
    print session.run(predictedOutputY)

# Plot the output using TensorBoard below, and visualize under Events
summaryPlot = tf.scalar_summary('predictedOutput', predictedOutputY)

# It's hard to keep track of your computational graph
# Therefore, can use TensorBoard, which is a visualization tool.
print "Visualizing Outputs and Computational Graph"
# Creates a directory called logSimpleGraph to store the current graph
summaryWriter = tf.train.SummaryWriter('logSimpleGraph', session.graph)

# Reset all the variables to initial values
session.run(tf.initialize_all_variables())
for i in range(100):
    summaryStr = session.run(summaryPlot)
    summaryWriter.add_summary(summaryStr, i)
    session.run(trainStepFunction)

print "To visualize Tensorboard, execute on commandline:"
# It will execute the visualization tool as a webserver for a browser
print ">> tensorboard --logdir=logSimpleGraph"
