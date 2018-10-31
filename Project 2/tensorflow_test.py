import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

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
X = np.concatenate((X_critical,X))
Y = np.concatenate((Y_critical,Y))

train_to_test_ratio = 0.8
#Splitting train(80), test(20)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_to_test_ratio)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(100,activation='sigmoid',input_dim=X.shape[1]))
#model.add(tf.keras.layers.Dropout(0.3))
#model.add(tf.keras.layers.Dense(50,activation='tanh'))
#model.add(tf.keras.layers.Dropout(0.3))
#model.add(tf.keras.layers.Dense(10,activation='tanh'))
#model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#print(model.summary())

model.fit(X_train,Y_train,epochs = 10,batch_size = 1000,validation_data=[X_test,Y_test])

print('accuracy: ',model.evaluate(X_test,Y_test))
