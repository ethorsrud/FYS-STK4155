import numpy as np
import scipy.sparse as sp
#from Project2_functions import *
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import os
import glob
import pandas as pd
import seaborn as sns
import sklearn.model_selection as skms
import sklearn.linear_model as skl
import sklearn.metrics as skm
import tqdm
import copy
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from NeuralNetwork import NeuralNetwork

np.random.seed(12)

def lambda_MSE_plot(lambdas,MSE):
    fig = plt.figure(figsize=(9,4))
    plt.plot(lambdas,MSE)
    plt.xlabel("Lambda")
    plt.ylabel("MSE")
    plt.show()

def OLS(X,y):
    U,S,V_T = scipy.linalg.svd(X,full_matrices=False)
    smat = np.zeros((len(S),len(S)))
    smatinv = np.zeros((len(S),len(S)))
    for i,val in enumerate(S):
        if val<= 1e-15*np.max(S):
            smat[i,i] = 0
            smatinv[i,i] = 0
        else:
            smat[i,i] = val
            smatinv[i,i] = 1/val

    beta = V_T.T@smatinv@U.T@y

    y_fit = X@beta

    MSE = np.mean((y-y_fit)**2)
    R_2 = 1-(np.sum((y-y_fit)**2))/(np.sum((y-np.mean(y))**2))
    var_OLS = np.mean((y_fit-np.mean(y_fit))**2)
    bias_OLS = np.mean((y-np.mean(y_fit))**2)
    print ("MSE OLS = ",MSE)
    print ("R2 OLS = ",R_2)
    #print ("Bias OLS ",bias_OLS)
    print ("Variance OLS", var_OLS)
    return y_fit, beta

def Ridge(X,y,lambd):
    preds = len(X[0,:])
    beta_Ridge = np.dot(np.linalg.inv(np.dot(X.T,X)+lambd*np.identity(preds)),np.dot(X.T,y))
    y_fit_ridge = np.dot(X,beta_Ridge)
    MSE_ridge = np.mean((y-y_fit_ridge)**2)
    R_2_ridge = 1-(np.sum((y-y_fit_ridge)**2))/(np.sum((y-np.mean(y))**2))
    var_ridge = np.mean((y_fit_ridge-np.mean(y_fit_ridge))**2)
    bias_ridge = np.mean((y-np.mean(y_fit_ridge))**2)
    print ("-------lambda = %f-------"%lambd)
    print ("MSE Ridge = ",MSE_ridge)
    print ("R2 Ridge = ",R_2_ridge)
    #print ("Bias_ridge = ",bias_ridge)
    print ("Variance ridge = ",var_ridge)
    return y_fit_ridge, beta_Ridge

def Lasso(X,y,lambd):
    lasso = linear_model.Lasso(alpha=lambd,fit_intercept=True,max_iter=int(10**3))
    lasso.fit(X,y)
    beta_lasso = lasso.coef_
    y_fit_lasso = np.dot(X,beta_lasso)
    MSE_lasso = np.mean((y-y_fit_lasso)**2)
    R_2_lasso = 1-(np.sum((y-y_fit_lasso)**2))/(np.sum((y-np.mean(y))**2))
    var_lasso = np.mean((y_fit_lasso-np.mean(y_fit_lasso))**2)
    bias_lasso = np.mean((y-np.mean(y_fit_lasso))**2)
    print( "-------lambda = %f-------"%lambd)
    print ("MSE Lasso = ",MSE_lasso)
    print ("R2 Lasso = ",R_2_lasso)
    #print ("Bias lasso = ",bias_lasso)
    print ("Variance lasso = ",var_lasso)
    return y_fit_lasso,beta_lasso

def kfold(X,y,k,type,lambd):
    m = len(X[:,0])
    preds = len(X[0,:]) #Number of predictors
    z2 = np.zeros(shape=(int(m/k),k)) #array of test data
    z2_fit = np.zeros(shape=(int(m/k),k)) #Fitted test data
    #MSE_array = np.zeros(shape=(len(x.flatten())/k,1))
    for i in range(int(m/k)):
        #Training set with x,y,z[everything up to i*k and everything after k+i*k]
        X_train = np.concatenate((X[:(i*k)],X[(k+i*k):]))
        y_train = np.concatenate((y[:(i*k)],y[(k+i*k):]))
        #Testing set with x,y,z[everything between i*k and k+i*k]
        X_test = X[(i*k):(k+i*k)]
        y_test = y[(i*k):(k+i*k)]
        if(type=="Ridge"):
            beta = np.dot(np.linalg.inv(np.dot(X_train.T,X_train)+lambd*np.identity(preds)),np.dot(X_train.T,y_train))
        elif(type=="Lasso"):
            lasso = linear_model.Lasso(alpha=lambd,fit_intercept = True,max_iter=int(10**5),tol=0.0001)
            lasso.fit(X_train,y_train)
            beta = lasso.coef_
        else:
            U,S,V_T = np.linalg.svd(X_train,full_matrices=False)
            S = np.diag(S)
            beta = np.dot(V_T.T,np.dot(np.linalg.inv(S),np.dot(U.T,y_train)))
        z2_fit[i] = np.dot(X_test,beta) #+ np.mean(z) #Adding mean for centered data
        z2[i] = y_test #+ np.mean(z) ##Adding mean for centered data
        #Terminal progress bar
        sys.stdout.write("\r kfold progress: %.2f%%"%(float(i+1)*100/(m/k)))
        sys.stdout.flush()

    #bias_kfold = np.mean((z2-np.mean(z2_fit))**2)
    MSE_k_fold = np.mean((z2-z2_fit)**2)
    R2_k_fold = 1-(np.sum((z2-z2_fit)**2))/(np.sum((z2-np.sum(z2)/(m))**2))
    var_kfold = np.mean((z2_fit-np.mean(z2_fit))**2)
    #e_term (extra term) for comparing MSE = bias + variance + extra term
    #e_term = (2./m)*np.sum((z2-np.mean(z2_fit))*(np.mean(z2_fit)-z2_fit))

    print ("-------------------------")
    print ("MSE_kfold %s = "%type,MSE_k_fold)
    print ("R2_kfold %s ="%type,R2_k_fold)
    #print ("Bias kfold ", bias_kfold)
    print ("var k-fold ", var_kfold)
    #print "------------------------"
    return MSE_k_fold,R2_k_fold

def OLS_Lasso_Ridge_1D(X,y,lambdas,L):
    for lambd in lambdas:
        cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')
        y_fit,beta_OLS = OLS(X,y)
        y_fit_ridge,beta_ridge = Ridge(X,y,lambd)
        y_fit_lasso,beta_lasso = Lasso(X,y,lambd)
        J_ridge = beta_ridge.reshape((L,L))
        J_lasso = beta_lasso.reshape((L,L))
        J_OLS = beta_OLS.reshape((L,L))
        fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(15,4))
        ax[0].imshow(J_OLS,**cmap_args)
        ax[0].set_title('$\\mathrm{OLS}$',fontsize=16)
        ax[0].tick_params(labelsize=16)

        ax[1].imshow(J_ridge,**cmap_args)
        ax[1].set_title('$\\mathrm{Ridge},\ \\lambda = %.4f$'%lambd,fontsize=16)
        ax[1].tick_params(labelsize=16)

        im = ax[2].imshow(J_lasso,**cmap_args)
        ax[2].set_title('$\\mathrm{Lasso},\ \\lambda = %.4f$'%lambd,fontsize=16)
        ax[2].tick_params(labelsize=16)
        div = make_axes_locatable(ax[2])
        cax = div.append_axes("right",size="5%",pad = 0.05)
        cbar = fig.colorbar(im,cax=cax)
        cbar.ax.set_yticklabels(np.arange(-1.0,1.0+0.25,0.25),fontsize=12)
        cbar.set_label('$J_{i,j}$',labelpad=-40,y=1.12,fontsize=16,rotation=0)

        fig.subplots_adjust(right=0.85)
        plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def logistic_cost(h,y):
    return np.mean(-y*np.log(h)-(1-y)*np.log(1-h))

def pred(X,beta,threshold):
    return sigmoid(X@beta)>=threshold

def logisticregression(X,y,epochs,batch_size):
    np.random.seed(1)
    t0 = 1
    t1 = 10
    iter = int(len(X[:,0])/batch_size)
    beta = np.zeros(X.shape[1])
    for k in range(epochs):
        for i in range(iter):
            rand_batch = np.random.choice(len(X[:,0]),size=batch_size)#np.random.choice(len(X[:,0]),size=(1,batch_size),replace=False)[0]

            X_batch = X[rand_batch]
            y_batch = y[rand_batch]

            y_fit = X_batch@beta
            h = sigmoid(y_fit)
            gradient = -X_batch.T@(y_batch-h)
            beta -= (t0/(k*iter+i+t1))*gradient
    return beta






import warnings
warnings.filterwarnings('ignore')

L = 40
states = np.random.choice([-1,1],size=(10000,L))

def ising_energies(states,L):
    J = np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    E = np.einsum('...i,ij,...j->...',states,J,states)
    return E

energies = ising_energies(states,L)

states = np.einsum('...i,...j->...ij',states,states)
shape = states.shape
states = states.reshape((shape[0],shape[1]*shape[2]))

data = [states,energies]

X = data[0]
Y = data[1]

train_to_test_ratio = 0.8
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_to_test_ratio)

y_fit_train, beta_train = OLS(X_train,Y_train)
y_fit_test = X_test@beta_train

print("------OLS------")
print("MSE test:",np.mean((Y_test-y_fit_test)**2))
print("R2 test: ",1-(np.sum((Y_test-y_fit_test)**2))/(np.sum((Y_test-np.mean(Y_test))**2))
)
OLS_Lasso_Ridge_1D(X_train,Y_train,lambdas,L)


Y = (Y-np.min(Y))/(np.max(Y)-np.min(Y)) #normalizing
#Splitting train(80), test(20)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_to_test_ratio)

print("------NN1D------")
#From NeuralNetwork.py
NN1D = NeuralNetwork(n_nodes=[100],n_outputs=1,batch_size=10,epochs=10,learningrate=0.1,cost="Ordinary") #n_hidden,n_nodes,n_outputs,batch_size,epochs,learningrate):
NN1D.run(X_train,Y_train)
print("R2 test: ",NN1D.R2(X_test,Y_test))


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

X_O_train,X_O_test,Y_O_train,Y_O_test=train_test_split(X_ordered,Y_ordered,train_size=train_to_test_ratio)

X_critical=data[70000:100000,:]
Y_critical=labels[70000:100000]

X_C_train,X_C_test,Y_C_train,Y_C_test=train_test_split(X_critical,Y_critical,train_size=train_to_test_ratio)

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]

X_D_train,X_D_test,Y_D_train,Y_D_test=train_test_split(X_disordered,Y_disordered,train_size=train_to_test_ratio)

X = np.concatenate((X_ordered,X_disordered))
Y = np.concatenate((Y_ordered,Y_disordered))
#Adding Critical data
X_all = np.concatenate((X_critical,X))
Y_all = np.concatenate((Y_critical,Y))


#Splitting train(80), test(20)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_to_test_ratio)
X_train_all,X_test_all,Y_train_all,Y_test_all=train_test_split(X_all,Y_all,train_size=train_to_test_ratio)


#Neural network 2D
print("------NN2D------")
#From NeuralNetwork.py
NN2D = NeuralNetwork(n_nodes=[100],n_outputs=1,batch_size=1000,epochs=10,learningrate=0.00001,cost="Classification") #n_hidden,n_nodes,n_outputs,batch_size,epochs,learningrate):
NN2D.run(X_train,Y_train)

p = np.linspace(0.1,0.9,20)
accuracies = np.zeros(20)
for i in range(len(p)):
    print ("Accuracy test(Ordered+Disordered) p = %f:  "%p[i],NN2D.accuracy(X_test,Y_test,p[i]))
    accuracies[i] = NN2D.accuracy(X_test,Y_test,p[i])
plt.plot(p,accuracies)

print("Accuracy ordered",NN2D.accuracy(X_O_test,Y_O_test,0.5))
print("Accuracy Disordered",NN2D.accuracy(X_D_test,Y_D_test,0.5))
print("Accuracy Critical",NN2D.accuracy(X_C_test,Y_C_test,0.5))
print("Accuracy Test",NN2D.accuracy(X_test_all,Y_test_all,0.5))
print("Accuracy Train",NN2D.accuracy(X_train,Y_train,0.5))

#Logistic regression 2D
print("------LOGREG2D------")
beta = logisticregression(X_train,Y_train,10,10) #X,y,epochs,batch_size
p = np.linspace(0.1,0.9,20)
accuracies = np.zeros(20)

for i in range(len(p)):
    print ("Accuracy test(Ordered+Disordered) p = %f:  "%p[i],np.mean(pred(X_test,beta,p[i])==Y_test))
    accuracies[i] = np.mean(pred(X_test,beta,p[i])==Y_test)
plt.plot(p,accuracies)
print("Accuracy ordered",np.mean(pred(X_O_test,beta,0.46)==Y_O_test))
print("Accuracy Disordered",np.mean(pred(X_D_test,beta,0.46)==Y_D_test))
print("Accuracy Critical",np.mean(pred(X_C_test,beta,0.46)==Y_C_test))
print("Accuracy Test",np.mean(pred(X_test_all,beta,0.46)==Y_test_all))
print("Accuracy Train",np.mean(pred(X_train,beta,0.46)==Y_train))

plt.xlabel("Probability threshold - p")
plt.ylabel("Classification Accuracy")
plt.legend(["Test - Neural network","Test - Logistic reg."])

plt.show()

print("------SKLEARN------")

clf = LogisticRegression(random_state=0,solver='sag').fit(X_train,Y_train)
print ("SKLEARN Train accuracy(Ordered+Disordered)",clf.score(X_train,Y_train))
print ("SKLEARN Test accuracy(Ordered+Disordered)",clf.score(X_test,Y_test))
