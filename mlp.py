import numpy as np

def tanh(x):
	return (np.exp(2*x)-1)/(np.exp(2*x)+1)

def tanh_der(x):
	return 4/((np.exp(x)+np.exp(-x))**2)

number_of_neurons = int(input("Please, enter the number of neurons "))
number_of_layers = int(input("Please, enter the number of layers ")) 

class NN:
    def __init__(self, inputs):
        self.inputs = inputs
        self.l = len(self.inputs)
        self.li = len(self.inputs[0])

        self.wi = np.random.random((self.li, self.l))
        self.wh = np.random.random((self.l, 1))

        

    def generate_weights(self):
        self.weights = [np.random.uniform(0,1) for i in range(0, self.lenght)]

    def get_weights(self):
        print(self.weights)
        
    def think(self, inp):
        s1 = tanh(np.dot(inp, self.wi))
        s2 = tanh(np.dot(s1, self.wh))
        return s2

    def train(self, inputs,outputs, it):
        for i in range(it):
            l0 = inputs
            l1 = tanh(np.dot(l0, self.wi))
            l2 = tanh(np.dot(l1, self.wh))

            l2_err = outputs - l2
            l2_delta = np.multiply(l2_err, tanh_der(l2))

            l1_err = np.dot(l2_delta, self.wh.T)
            l1_delta = np.multiply(l1_err, tanh_der(l1))

            self.wh+= np.dot(l1.T, l2_delta)
            self.wi+= np.dot(l0.T, l1_delta)
            
           

    def backpropagate(self, target, lrate=0.1):
        
        deltas = []

        error = target - self.layers[-1]
        delta = error*tanh_der(self.layers[-1])
        deltas.append(delta)

        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*tanh_der(self.layers[i])
            deltas.insert(0,delta)
            
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw

        
        return (error**2).sum()

inputs=np.array([[0,0], [0,1], [1,0], [1,1] ])
outputs=np.array([[0,1,1,0]]).T

n=NN(inputs)
print(n.think(inputs))
n.train(inputs, outputs, 20000)
a = (n.think(inputs))
a[a > 0.5] = 1
a[a < 0.5] = 0
print(a)

