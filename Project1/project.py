from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from project1_functions import *
from sklearn import linear_model
from imageio import imread
import scipy


np.random.seed(1)

#----General settings----
n = 20
m = n*n
p = 5 #Polynomial order
polys = sum(range(1,p+2)) #Number of polynomials
sigma = 0.1
x = np.linspace(0,1,n)
y = np.linspace(0,1,n)
x, y = np.meshgrid(x,y)
X = designmatrix(x.flatten(),y.flatten(),p)
H = np.dot(np.transpose(X),X)

#----OLS without k-fold----
z = FrankeFunction(x, y)
z += np.random.randn(n,n)*sigma #Adding noise
z_fit,beta_OLS,MSE,R_2,bias_OLS,var_OLS = OLS(z,H,X)
conf_sigma = np.sum((z.flatten()-z_fit.flatten())**2)/(m-polys) #sigma for calculating confidence interval
Confidence_plot(beta_OLS,conf_sigma,X,H,polys,"OLS",0) #(beta,sigma,X,H,polys,type,lambd)

#----OLS with k-fold----
kfold(x,y,z,5,"OLS",0.0,5) #(x,y,k,type,lambda,p)

#----Ridge without k-fold----
np.random.seed(1) #Resetting seed to reproduce the same as for OLS
z = FrankeFunction(x,y)
z += np.random.randn(n,n)*sigma #Adding noise
X = designmatrix(x.flatten(),y.flatten(),p)
H = np.dot(np.transpose(X),X)

#Finding the optimal lambda
lambdas = np.linspace(0,0.0000001,15)
MSE_ridge = np.zeros(15)
for i in range(len(lambdas)):
    MSE_ridge[i] = Ridge(z,H,X,lambdas[i],polys)[2]
lambda_opt = lambdas[np.argmin(MSE_ridge)]
#Ridge regression with the optimal lambda
z_fit_ridge,beta_Ridge,MSE_ridge,R_2_ridge,bias_ridge,var_ridge = Ridge(z,H,X,lambda_opt,polys)
conf_sigma = np.sum((z.flatten()-z_fit_ridge.flatten())**2)/(m-polys) #sigma for calculating confidence interval
Confidence_plot(beta_Ridge,conf_sigma,X,H,polys,"Ridge",lambda_opt) #(beta,sigma,X,H,polys,type,lambd)

#----Ridge with k-fold----
lambdas = np.linspace(0,0.0004,15)
MSE_ridge = np.zeros(15)
R2_ridge = np.zeros(15)

for i in range(len(lambdas)):
    MSE_ridge[i],R2_ridge[i] = kfold(x,y,z,5,"Ridge",lambdas[i],5) #(x,y,z,k,type,lambd,p):
lambda_MSE_plot(lambdas,MSE_ridge)

#----Checking Optimal Lambda for different noises----
sigmas = np.linspace(0,0.27,6)
lambda_optimals = np.zeros(6)
j = 0
for value in sigmas:
    np.random.seed(1)
    z = FrankeFunction(x,y)
    sigma = value
    z += np.random.randn(n,n)*sigma #Adding noise
    for i in range(len(lambdas)):
        MSE_ridge[i],R2_ridge[i] = kfold(x,y,z,5,"Ridge",lambdas[i],5) #def kfold(x,y,z,k,type,lambd,p):
    lambda_optimals[j] = lambdas[np.argmin(MSE_ridge)]
    j+=1
plt.plot(sigmas**2,lambda_optimals)
plt.xlabel("Sigma**2")
plt.ylabel("Lambda_optimal")
plt.show()

#----lasso without k-fold-----
np.random.seed(1) #Resetting seed to reproduce the same as for OLS
z = FrankeFunction(x,y)
z += np.random.randn(n,n)*sigma #Adding noise
#Finding the optimal lambda
lambdas = np.linspace(0,0.000002,10)
MSE_lasso = np.zeros(10)
for i in range(len(lambdas)):
    MSE_lasso[i] = Lasso(z,H,X,lambdas[i])[2]
lambda_opt = lambdas[np.argmin(MSE_lasso)]
#Lasso regression with the optimal lambda
z_fit_lasso,beta_Lasso,MSE_lasso,R_2_lasso,bias_lasso,var_lasso = Lasso(z,H,X,lambda_opt)


#----Lasso with k-fold----
lambdas = np.linspace(0,0.000002,10)
MSE_lasso = np.zeros(10)
R2_lasso = np.zeros(10)
for i in range(len(lambdas)):
    MSE_lasso[i],R2_lasso[i] = kfold(x,y,z,5,"Lasso",lambdas[i],5) #def kfold(x,y,z,k,type,lambd,p):

lambda_MSE_plot(lambdas,MSE_lasso)


#Real data
# Load the terrain
terrain1 = imread('SRTM_data_Norway_1.tif')
terrain1 = terrain1.astype(float)
#----General settings----
pixelx = 50
pixely = 50
p = 5 #Polynomial order
polys = sum(range(1,p+2)) #Number of polynomials
z = np.array(terrain1[:pixelx,:pixely],dtype=float).flatten()
x = np.linspace(0,1,pixelx)
y = np.linspace(0,1,pixely)
z = (z-np.min(z))/(np.max(z)-np.min(z)) #normalizing

x,y = np.meshgrid(x,y)
m = len(x.flatten())
X = designmatrix(x.flatten(),y.flatten(),p)
H = np.dot(X.T,X)

#----Ordinary least squares----
z_fit,beta_OLS,MSE,R_2,bias_OLS,var_OLS = OLS(z,H,X)
conf_sigma = np.sum((z.flatten()-z_fit.flatten())**2)/(m-polys)
Confidence_plot(beta_OLS,conf_sigma,X,H,polys,"OLS",0)

#----kfold----
kfold(x,y,z.reshape(pixelx,pixely),5,"OLS",0,p,0) #kfold(x,y,z,k,type,lambd,p):
#surfplot(x,y,z.reshape(pixelx,pixely))

#----Ridge----
#Finding the optimal lambda for Ridge
lambdas = np.linspace(0,0.0000000001,15)
MSE_ridge = np.zeros(15)
for i in range(len(lambdas)):
    MSE_ridge[i] = Ridge(z,H,X,lambdas[i],polys)[2]
lambda_opt = lambdas[np.argmin(MSE_ridge)]
print "Lambda Optimal " ,lambda_opt
#Ridge regression with the optimal lambda
z_fit_ridge,beta_Ridge,MSE_ridge,R_2_ridge,bias_ridge,var_ridge = Ridge(z,H,X,lambda_opt,polys)
conf_sigma = np.sum((z.flatten()-z_fit_ridge.flatten())**2)/(m-polys) #sigma for calculating confidence interval
Confidence_plot(beta_Ridge,conf_sigma,X,H,polys,"Ridge",lambda_opt) #(beta,sigma,X,H,polys,type,lambd)
#surfplot(x,y,z_fit_ridge.reshape(pixelx,pixely))

#----Ridge k-fold----
lambdas = np.linspace(0,0.00004,10)
MSE_ridge = np.zeros(10)
R2_ridge = np.zeros(10)
for i in range(len(lambdas)):
    MSE_ridge[i],R2_ridge[i] = kfold(x,y,z.reshape(pixelx,pixely),5,"Ridge",lambdas[i],p) #def kfold(x,y,z,k,type,lambd,p):
print "Lambda: ", lambdas[np.argmin(MSE_ridge)]
print "MSE Ridge: ", MSE_ridge[np.argmin(MSE_ridge)]
lambda_MSE_plot(lambdas,MSE_ridge)

#Finding the optimal lambda for Lasso
lambdas = np.linspace(0,0.000002,10)
MSE_lasso = np.zeros(10)
for i in range(len(lambdas)):
    MSE_lasso[i] = Lasso(z,H,X,lambdas[i])[2]
lambda_opt = lambdas[np.argmin(MSE_lasso)]
#Lasso regression with the optimal lambda
z_fit_lasso,beta_Lasso,MSE_lasso,R_2_lasso,bias_lasso,var_lasso = Lasso(z,H,X,lambda_opt)
#surfplot(x,y,z_fit_lasso.reshape(pixelx,pixely))

#----Lasso k-fold----
lambdas = np.linspace(0,0.0000005,10)
MSE_lasso = np.zeros(10)
R2_lasso = np.zeros(10)
for i in range(len(lambdas)):
    MSE_lasso[i],R2_lasso[i] = kfold(x,y,z.reshape(pixelx,pixely),5,"Lasso",lambdas[i],p)
lambda_MSE_plot(lambdas,MSE_lasso)
print "Lambda: ", lambdas[np.argmin(MSE_lasso)]
print "MSE Lasso: ", MSE_lasso[np.argmin(MSE_lasso)]
