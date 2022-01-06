import random as r
from cv2 import add, resize
import networkx as nx
import matplotlib.pyplot as plt 
import time

from numpy.core.fromnumeric import product

class Innovation():
    def __init__(self, initial) -> None:
        '''
        Just handels handing out innovation nubers so it is gloabally shared in the population
        '''
        self.number = initial

    def getValue(self):
        self.number += 1
        return self.number

class Node():
    def __init__(self, layer, innovation_number, default_output, operation):
        '''
        For feed forward:
        set all nodes updated to false
        set first notes output values, set their updated to true
        call getOutput on only final nodes
        '''

        self.inputs = {}
        self.layer = layer
        self.innovation_number = innovation_number
        self.default_output = default_output
        self.output = default_output
        self.updated = False
        self.operation = operation


    def addInputs(self, inputInnovation, connectionInnovation):
        inputInnovations = []
        for indexs in self.inputs.keys():
            inputInnovations.append(self.inputs[indexs])
        if inputInnovation not in inputInnovations:
            self.inputs[connectionInnovation] = inputInnovation

    def getInnovation(self):
        return self.innovation_number

    def removeLostConnections(self, model):
        '''
        Remove connections that are dangeling
        Used to point to a node which has since been removed
        '''
        remove = []
        for connection in self.inputs.keys():
            if not self.inputs[connection] in model.keys():
                remove.append(connection)
        for value in remove:
            del self.inputs[value]

    def updateOutput(self, model):
        self.removeLostConnections(model)
        values = []
        for connection in self.inputs.keys():
            values.append(model[self.inputs[connection]].getOutput(model))
        if len(values) == 0:
            self.output =  self.default_output
        else:
            self.output = self.operation(values) 

    def getOutput(self, model):
        if self.updated == False:
            self.updateOutput(model)
            self.updated = True
        return self.output

    def getLayer(self):
        return self.layer

class AgentFFO():
    def __init__(self, inputs, outputs, max_layers, default_output, operations, 
                    add_node_rate, add_connection_rate, remove_node_rate, remove_connection_rate):
        '''
        Feed Forward Only Agent
        '''
        self.inputs = inputs 
        self.outputs = outputs
        self.max_layers = max_layers
        self.default_output = default_output
        self.operations = operations 
        self.add_node_rate = add_node_rate
        self.add_connection_rate = add_connection_rate 
        self.remove_connection_rate = remove_connection_rate
        self.remove_node_rate = remove_node_rate

        self.initModel()

    def initModel(self):
        self.inputsInnovation = list(range(self.inputs))
        self.outputsInnovation = list(range(self.inputs, self.inputs+self.outputs))
        self.model = {}

        for x in self.inputsInnovation:
            self.model[x] = Node(-1,x,self.default_output, r.choice(self.operations))
        for x in self.outputsInnovation:
            self.model[x] = Node(self.max_layers,x,self.default_output, r.choice(self.operations))

    def forward(self, input):
        assert len(input) == self.inputs

        outputs = []

        for key in self.model.keys():
            self.model[key].updated = False
        for x in self.inputsInnovation:
            self.model[x].output = input[x]
            self.model[x].updated = True
        for x in self.outputsInnovation:
            outputs.append(self.model[x].getOutput(self.model))

        return outputs

    def mutate(self, inno):
        # Add connections
        addedConnection = False or r.random() > self.add_connection_rate
        nodes = list(self.model.keys())
        while not addedConnection:
            output = r.choice(nodes)
            inputNode = r.choice(nodes)
            if self.model[output].layer > self.model[inputNode].layer:
                self.model[output].addInputs(inputNode, inno.getValue())
                addedConnection = True

        # Add node
        addNode = r.random() < self.add_node_rate
        if addNode:
            layer = r.randint(0,self.max_layers-1)
            self.model[inno.getValue()] = Node(layer,inno.getValue(),self.default_output, r.choice(self.operations))

        # Remove node
        if r.random() < self.remove_connection_rate and len(list(self.model.keys())) > self.inputs + self.outputs:
            nodeToRemove = r.choice(list(self.model.keys()))
            while nodeToRemove <= self.inputs + self.outputs:
                nodeToRemove = r.choice(list(self.model.keys()))
            del self.model[nodeToRemove]

            for node in self.model.keys():
                if nodeToRemove in self.model[node].inputs.values():
                    del self.model[node].inputs[list(self.model[node].inputs.keys())[list(self.model[node].inputs.values()).index(nodeToRemove)]]
        
        # Remove Connection
        if r.random() < self.remove_node_rate:
            nodeToRemove = r.choice(list(self.model.keys()))
            x = 0
            while (nodeToRemove <= self.inputs or len(list(self.model[nodeToRemove].inputs.keys())) == 0) and x < 25:
                nodeToRemove = r.choice(list(self.model.keys()))
                x+= 1

            if x < 25:
                connectiontoRemove = r.choice(list(self.model[nodeToRemove].inputs.keys()))
                del self.model[nodeToRemove].inputs[connectiontoRemove]
    
        # get rid of dangeling connections:
        for node in self.model.keys():
            self.model[node].removeLostConnections(self.model)

    def crossoverModels(self, secondModel):
        '''
        Crosses over or takes the union of two models.
        Does not modify the referanced model.
        If applied where the target model just initalized makes a copy of the referance model. 
        '''
        myKeys = list(self.model.keys())
        theirKeys = (secondModel.model.keys())

        nodeIntersection = [value for value in myKeys if value in theirKeys] 

        # add the connections
        for node in nodeIntersection:
            self.model[node].inputs = {**secondModel.model[node].inputs, **self.model[node].inputs }
        
        self.model = {**secondModel.model, **self.model}

    def VisualizeModel(self):
        G = nx.MultiDiGraph()

        edges = []

        nodes = list(self.model.keys())
        
        for node in nodes:
            G.add_node(node, layer=self.model[node].getLayer())

        for node in nodes:
            for input in self.model[node].inputs.keys():
                edges.append((self.model[node].inputs[input], node))
        G.add_edges_from(edges)

        plt.figure(figsize=(8,8))

        pos = nx.multipartite_layout(G, subset_key="layer")

        nx.draw(G, pos, connectionstyle='arc3, rad = 0.1', with_labels = True)

        plt.show()

    def getGenome(self):
        nodeInnovations = list(self.model.keys())[self.inputs+self.outputs-1:]
        connectionInnovations = []
        for nodeInno in nodeInnovations:
            connectionInnovations += list(self.model[nodeInno].inputs.keys())
        return nodeInnovations + connectionInnovations

if __name__ == '__main__':
    
    model = AgentFFO(8, 4, 4, 0, [sum], 0.25, 0.5, 0.1, 0.1)
    model1 = AgentFFO(8, 4, 4, 0, [sum], 0.25, 0.5, 0.1, 0.1)

    inno = Innovation(8+4)
    
    for x in range(25):
        model.mutate(inno)
        model1.mutate(inno)

    model.VisualizeModel()

    model1.VisualizeModel()

    model.crossoverModels(model1)

    model.VisualizeModel()

    start = time.time()
    result = model.forward([1,1,1,1,1,1,1,1])
    print(abs(start-time.time()))
    print(result)
