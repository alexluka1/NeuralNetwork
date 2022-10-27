import math
#import matplotlib.pyplot as plt


## Need to have the files uploaded to colab for this to work
trainData = [] # Creating list for train data
testData = [] # Creating list for test data

## Train data
with open ("data-CMP2020M-item1-train.txt", "r") as trainFile: # Opens file in read mode
  for line in trainFile: # Loops through every line of the file 
    line = line.split() # Splits the line of the file by their spaces
    # Looping through each index of list created from the line
    for index in range(len(line)):
      line[index] = float(line[index]) # Turning each element of line into a float
    trainData.append(line) # Appening data to list
  trainFile.close() #Closes the file

## Test data
with open("data-CMP2020M-item1-test.txt", "r") as testFile: # Opens file in read mode
  for line in testFile: # Loops through every line of the file 
    line = line.split() # Splits the line of the file by their spaces
    # Looping through each index of list created from the line
    for index in range(len(line)-2): # -2 is added as last two elements are ('?') unknown
      line[index] = float(line[index]) # Turning each element of line into a float
    testData.append(line) # Appening data to list
  testFile.close() #Closes the file



print("Train data")
for i in trainData: 
  print(i)
print("\n\nTest data")
for i in testData:
  print(i)


class mathClass():
    @staticmethod
    def sigmoid(net):
      calc = 1/(1 + (math.exp(-net)))
      return calc

    @staticmethod
    def softmax(vector, t):
      bottom = 0
      for v in vector:
        bottom += math.exp(v)
      return math.exp(t)/bottom


class nodeClass():
  def __init__(self, _name):
    self.output = 0.0 # Set as a defult value
    self.input = 0.0
    self.name = _name

  def __repr__(self):
    return str(self.name)

  def calcOutput(self):
    self.output = mathClass.sigmoid(self.input)


class layerClass():
  def __init__(self, _name):
    self.nodes = []
    self.noNodes = 0
    self.name = _name

  def addNode(self, node):
    self.nodes.append(node)
    self.noNodes += 1

  ## Tells us if a node is in a layer
  def contains(self, input):
    for node in self.nodes:
      if node.name == input:
        return True
    return False


class targetClass():
  def __init__(self, _nodePair, _value):
    self.nodePair = _nodePair
    self.value = _value


class networkClass():
  def __init__(self):
    self.layers = []
    self.allNodes = []
    # All defult weights in the weightDict
    self.weightDict = {'w14':0.74, 'w15':0.13, 'w16':0.68, 'w24':0.8, 'w25':0.4, 'w26':0.10, 'w34':0.35, 'w35':0.97, 'w36':0.96, 'w47':0.35, 'w48':0.8, 'w57':0.50, 'w58':0.13, 'w67':0.90, 'w68':0.8, 'w04':0.9, 'w05':0.45, 'w06':0.36, 'w07':0.98, 'w08':0.92,}

  def addLayer(self, layer):
    self.layers.append(layer)


  ## Function to complete forward step
  def forwardStep(self):
    #print(self.weightDict) ## Printing out the weight dict to fill in the table
    outputs = [] # Create list to store outputs
    ## For loop to loop through all nodes
    for node in range(len(self.allNodes)):
      #print(self.allNodes[node], self.allNodes[node].input) ## Does have inputs

      net = 0
      ## For loop to loop through all definition in dict
      for key in self.weightDict.keys():
        if (key[2] == str(self.allNodes[node].name)): ## If dict holds a weight going to the node add to net
          inputNode = int(key[1])
          inputValue = self.allNodes[inputNode].output
          net += inputValue * self.weightDict[key]

          #print("Node", inputNode, ":", inputValue, "*", self.weightDict[key], "=", net, "Bias node input:", self.allNodes[0].input, "Should be 1")

      ## Sets node input to net as long as its not in input layer
      if (self.allNodes[node].name >= self.layers[0].noNodes):
        self.allNodes[node].input = net
        #print("Node netted:", node)


      ## Gets output layer
      for layer in self.layers:
        if (layer.name == "output"):
          outL = layer


      ## If node is on output layer sigmoid func is not applied
      if (outL.contains(self.allNodes[node].name)):
        ## Sets the output of the output nodes = to their input
        self.allNodes[node].output = self.allNodes[node].input
        
        outputs.append(targetClass(node, self.allNodes[node].output))

        #print ("Network output", self.allNodes[node], "=", self.allNodes[node].output)

      else:
        ## Calcs output as long as its not in input layer
        if (self.allNodes[node].name >= self.layers[0].noNodes):
          ## Calculates the output of the node using the input and the sigmoid func
          self.allNodes[node].calcOutput()
        else:
          self.allNodes[node].output = self.allNodes[node].input
      #print(node, "input:", self.allNodes[node].input, "output:", self.allNodes[node].output)

    return outputs

      
  def backStep(self, targets, lr):
    ## Creating a func to create a list of the error values
    def createErrorList(self, targets):
      errorList = list(range(len(self.allNodes))) ## Creates list with n elements

      ## For loop to iterate down the list 
      for node in range(len(self.allNodes)-1,-1,-1):
        ## Gets output layer
        for layer in self.layers:
          if (layer.name == "output"):
            outL = layer

        ## If node is on output layer different method is used
        if (outL.contains(self.allNodes[node].name)):
          for target in targets:
            if (target.nodePair == self.allNodes[node].name):
              e = target.value - self.allNodes[node].output
              errorList[node] = e
        else: ## No need for else statement as key[1] would not hold node on output layer
          errorValue = 0
          for key in self.weightDict.keys():
            if (key[1] == str(self.allNodes[node].name)): ## If dict holds a weight going from the node
              ## Calculate the error value for the node
              ## This does error values ahead x weights
              errorValue += errorList[int(key[2])] * self.weightDict[key]
          ## This does output * (1-output) * prev error value
          errorValue = self.allNodes[node].output * (1 - self.allNodes[node].output) * errorValue
          errorList[node] = errorValue
      return errorList


    errorList = createErrorList(self, targets)
    #print(errorList)

    ## Calculating the MSE of this forward step
    ## This is where the MSE is suppose to be
    ret = self.meanSquaredError(errorList)

    ## Use error list to change weights
    for node in range(len(self.allNodes)):
      for key in self.weightDict.keys():
        weightChange = 0
        if (key[1] == str(self.allNodes[node].name)): ## If dict holds a weight going from the node
          #print(node, ":", key) # Checks node gets the right key
          ## Formula for the weight change
          # lr * error * output
          weightChange = lr * errorList[int(key[2])] * self.allNodes[node].output
          #print(key, ":", lr, "*", errorList[int(key[2])], "*", self.allNodes[node].output, "=", weightChange) ## Tests that weight changes seem to make sense
          ## Adds weight change onto the value of the dict key
          self.weightDict[key] +=  weightChange
      #print("\n\n")
    
    return ret
    
        
  def meanSquaredError(self, errorList):
    MSE = 0
    for layer in self.layers:
      if (layer.name == "output"):
        div = layer.noNodes
        for node in layer.nodes:
          squaredError = errorList[node.name] ** 2
          MSE += squaredError
    return MSE / div #/2


## Each full step gets 2 squared errors for node 7 and 8
## Add these values to a running sum for error of each node
## At the end of the epoch, do these running sums / 6
## Then add each running sum together to get one point
## Gets MSE over the epoch to plot






  ## Func to allow user to view the network
  def printNetwork(self): # Prints out all node objects in their positioning
    print("Network structure:")
    for layer in self.layers:
      print(layer.name + ":", layer.nodes)
  
  def printInputs(self):
    for node in self.allNodes:
      print(node, "Input:", node.input)

  ## Set orgiinal weights
  def setOrigin(self, i1, i2, i3):
    self.allNodes[0].input = 1
    self.allNodes[1].input = i1
    self.allNodes[2].input = i2
    self.allNodes[3].input = i3


  


   ## Func for if size of network was undecided
  ## Could use this func to create network of any size
  ## Loops through all layers connecting them to the next layer, densely
  # def connnectLayers(self):
  #   for layer in range(len(self.layers)):
  #     for leftNode in self.layers[layer]:
  #       if ((layer+1) < len(layers)):
  #         for rightNode in self.layers[layer+1]:
  #           weightDict[str("w" + str(leftNode) + str(rightNode))] = 0.0








## Defining hyper parameters
learningRate = 0.1
epochs = 11

## Defining important objects
network = networkClass()

## Creating nodes, layers and the network

## Setting up input layer
inputLayer = layerClass("input")
for nodeNo in range(4): # Sets up the input nodes
  node = nodeClass(nodeNo)
  inputLayer.addNode(node)
  network.allNodes.append(node)


network.addLayer(inputLayer) # Adds input layer to the network

## Setting up hidden layer
hiddenLayer = layerClass("hidden")
for nodeNo in range(4, 7):
  node = nodeClass(nodeNo)
  hiddenLayer.addNode(node)
  network.allNodes.append(node)
network.addLayer(hiddenLayer)


## Setting up output layer
outputLayer = layerClass("output")
for nodeNo in range(7,9):
  node = nodeClass(nodeNo)
  outputLayer.addNode(node)
  network.allNodes.append(node)
network.addLayer(outputLayer)



print("Nodes:", network.allNodes)
network.printNetwork()
print("\n")
print(network.weightDict)
print("\n\n")


## Code to show how dictionary can be used to get connections
print("All connections of node 5:")
for key in network.weightDict.keys():
  if (key[2] == "5"):
    print(key, network.weightDict[key])
  elif (key[1] == "5"):
    print(key, network.weightDict[key])

print("\n\n")

# Using this code if a node dosent have any connections going the way specsified,
# then there will be no output
# As below:
for key in network.weightDict.keys():
  if (key[1] == "7"):
    print(key, network.weightDict[key])
MSElist = []
epochList = list(range(epochs)) ## Creates list with n elements
for i in range(epochs):
  #print(i, ":", network.weightDict)
  ## Mean Squared Error over Epoch
  MSEoE = 0
  for data in trainData:
    ## Sets inputs of network
    network.setOrigin(data[0],data[1],data[2])

    ##  Does a forward step on these inputs to get an output
    outputs = network.forwardStep()

    ## printing outputs for error checking
    # for o in outputs:
    #   print(o.nodePair, "->", o.value)
    # print("\n\n")

    ## Sets targets outputs
    targets = [targetClass(7, data[3]), targetClass(8, data[4])]
    ## Does a backward step using the targets, and the values from forward step to change weights
    MSEoE += network.backStep(targets, learningRate)
  ## Could divide by 6 to get the average MSE per epoch but going to use a summation of the MSE instead
  MSEoE = MSEoE #/ 6 ## Divides by 6 to get the average MSE ove the epoch
  MSElist.append(MSEoE)


#makeGraph(epochList, MSElist)




## Looping through data on test data
for data in testData:
  network.setOrigin(data[0],data[1],data[2])
  output = network.forwardStep()

  s = []
  for o in output:
    s.append(o.value)

  for o in output:
    print(o.nodePair, ":", mathClass.softmax(s, o.value))




