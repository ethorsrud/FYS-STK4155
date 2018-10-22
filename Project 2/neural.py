import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def neural_activation(x):
    return 1/(1+np.exp(-x))

def neuralnetwork2D(X,y,x_test,y_test,batch_size,epochs): #minibatching
    np.random.seed(1)
    eta = 0.001
    n_inputs = len(X[0,:])
    n_samples = len(X[:,0])
    n_outputs = 1
    n_hidden1 = 100
    weights1 = np.random.randn(n_inputs,n_hidden1)
    biases1 = np.random.randn(n_hidden1)
    weights2 = np.random.randn(n_hidden1,n_outputs)
    biases2 = np.random.randn(n_outputs)
    for k in range(epochs):
        for j in range(int(n_samples/batch_size)):
            rand_batch = np.random.choice(len(X[:,0]),size=batch_size,replace=True)
            input = X[rand_batch]
            z1 = input@weights1+biases1
            z2 = z1@weights2+biases2
            a1 = neural_activation(z1)
            a2 = neural_activation(z2)

            target = y[rand_batch]
            outputerror = a2-target[np.newaxis].T
            outputerror2 = outputerror@weights2.T*(a1*(1-a1))

            weights2 -= eta*a1.T@outputerror
            weights1 -= eta*input.T@outputerror2

            biases2 -= eta*np.sum(outputerror)
            biases1 -= eta*np.sum(outputerror2)
        print("Epoch %i"%k)

    #Testing accuracy
    fit = np.zeros(len(x_test[:,0]))
    targets = np.zeros(len(x_test[:,0]))

    z1 = x_test@weights1+biases1
    z2 = z1@weights2+biases2

    a1 = neural_activation(z1)
    a2 = neural_activation(z2)

    fit = ((a2>=0.5).T)*1
    print(fit,y_test)
    print (np.mean(fit==y_test))



# load data
file_name = "Ising2DFM_reSample_L40_T=All.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
data = pickle.load(open("./"+file_name,'rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
data=data.astype('int')

file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
labels = pickle.load(open("./"+file_name,'rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

# divide data into ordered, critical and disordered
X_ordered=data[:70000,:]
Y_ordered=labels[:70000]

X_critical=data[70000:100000,:]
Y_critical=labels[70000:100000]

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]

X = np.concatenate((X_ordered,X_disordered))
Y = np.concatenate((Y_ordered,Y_disordered))
#Adding Critical data
#X = np.concatenate((X,X_critical))
#Y = np.concatenate((Y,Y_critical))

train_to_test_ratio = 0.8
#Splitting train(80), test(20)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_to_test_ratio)
neuralnetwork2D(X_train,Y_train,X_test,Y_test,1000,10)
