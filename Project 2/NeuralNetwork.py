import numpy as np
import matplotlib.pyplot as plt
import scipy

np.random.seed(1)

class NeuralNetwork():
    def __init__(self,n_nodes,n_outputs,batch_size,epochs,learningrate,cost):
        self.n_hidden = len(n_nodes)
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.epochs = epochs
        self.learningrate = learningrate
        self.n_outputs = n_outputs
        self.ordinary = False
        self.classification = False
        if(cost=="Ordinary"):
            self.ordinary = True
        elif(cost=="Classification"):
            self.classification = True
        else:
            print("No cost function with that name")


    def initialize(self,X,y):
        #Initializing weights, biases, z's and activations a for the model
        self.n_inputs = len(X[0,:])
        W = np.empty(self.n_hidden+1,dtype=object)
        b = np.empty(self.n_hidden+1,dtype=object)
        z = np.empty(self.n_hidden+1,dtype=object)
        a = np.empty(self.n_hidden+1,dtype=object)
        outputerror = np.empty(self.n_hidden+1,dtype=object)
        W[0] = np.random.randn(self.n_inputs,self.n_nodes[0])*(1/np.sqrt(self.n_inputs))
        b[0] = np.random.randn(self.n_nodes[0])
        for i in range(1,self.n_hidden):
            W[i] = np.random.randn(self.n_nodes[i-1],self.n_nodes[i])*(1/np.sqrt(self.n_nodes[i-1]))
            b[i] = np.random.randn(self.n_nodes[i])
        W[-1] = np.random.randn(self.n_nodes[-1],self.n_outputs)*(1/np.sqrt(self.n_nodes[-1]))
        b[-1] = np.random.randn(self.n_outputs)
        self.W = W
        self.b = b
        self.z = z
        self.a = a
        self.outputerror = outputerror

    def sigmoid(self,x):
        #Sigmoid activation function
        return 1/(1+np.exp(-x))

    def sigmoid_deriv(self,x):
        #Derivative of sigmoid activation function
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def feedforward(self,X_batch):
        #Calculating the activation functions
        self.z[0] = self.X_batch@self.W[0]+self.b[0]
        self.a[0] = self.sigmoid(self.z[0])
        for i in range(1,self.n_hidden+1):
            self.z[i] = self.a[i-1]@self.W[i]+self.b[i]
            self.a[i] = self.sigmoid(self.z[i])

    def deriv_costfunction(self,a):
        #Return the derivative of the chosen cost function
        if self.ordinary:
            return(a-self.y_batch[np.newaxis].T)
        if self.classification:
            return((a-self.y_batch[np.newaxis].T)/(a*(1-a)))


    def backpropagate(self):
        #Calculating the outputerrors
        self.outputerror[-1] = self.deriv_costfunction(self.a[-1])*self.sigmoid_deriv(self.z[-1])#(self.a[-1]-self.y_batch[np.newaxis].T)
        for i in reversed(range(self.n_hidden)):
            self.outputerror[i] = self.outputerror[i+1]@self.W[i+1].T*(self.a[i]*(1-self.a[i]))
        #Updating the weights
        for i in range(1,self.n_hidden+1):
            self.W[i] -= self.learningrate*self.a[i-1].T@self.outputerror[i]
            self.b[i] -= self.learningrate*np.sum(self.outputerror[i],axis=0)
        self.W[0] -= self.learningrate*self.X_batch.T@self.outputerror[0]
        self.b[0] -= self.learningrate*np.sum(self.outputerror[0],axis=0)

    def run(self,X,y):
        self.initialize(X,y)
        n_samples = len(X[:,0]) #number of inputs
        m = int(n_samples/self.batch_size) #Number of batches
        for k in range(self.epochs):
            print("Epoch %i"%k)
            for j in range(m):
                #Creating a random batch
                rand_batch = np.random.choice(n_samples,size=self.batch_size,replace = False)
                self.X_batch = X[rand_batch]
                self.y_batch = y[rand_batch]
                #Feed forward and back propagate random batch
                self.feedforward(self.X_batch)
                self.backpropagate()

    def accuracy(self,X,y,threshold):
        #Calculating accuracy
        z = X@self.W[0]+self.b[0]
        a = self.sigmoid(z)
        for i in range(1,self.n_hidden+1):
            a_new = self.sigmoid(a@self.W[i]+self.b[i])
            a = a_new
        self.fit = ((a>=threshold).T)*1

        return np.mean(self.fit==y)

    def R2(self,X,y):
        #Calculating R2 score
        z = X@self.W[0]+self.b[0]
        a = self.sigmoid(z)
        for i in range(1,self.n_hidden+1):
            a_new = self.sigmoid(a@self.W[i]+self.b[i])
            a = a_new
        fit = a.T
        return (1-(np.sum((y-fit)**2))/(np.sum((y-np.mean(y))**2)))
