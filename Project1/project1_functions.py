import numpy as np
from sklearn import linear_model
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def designmatrix(x,y,p):
    index = 0
    X = np.ndarray(shape=(len(x),sum(range(1,p+2)))) #shape = length x, 21 (if p = 5)
    for i in range(p+1):
        for j in range(p+1-i):
            X[:,index] = (x**i)*(y**j) #x**0*y**0,... , x**0*y**p, x**1*y**0, ... , x**1*y**p, ... , ..., x**p*y**p
            index+=1
    return X

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def Confidence_plot(beta,sigma,X,H,polys,type,lambd):
    if type == "OLS":
        v_j = np.sqrt(np.diag(np.linalg.inv(H)))*sigma #square root of the diagonal elements in inv(X.T X)
    else:
        v_j = np.dot(np.linalg.inv(H+lambd*np.identity(polys)),np.dot(X.T,np.dot(X,np.linalg.inv(H+lambd*np.identity(polys)))))
        v_j = sigma*np.sqrt(np.diag(v_j))
    fig = plt.figure(figsize=(8, 4))
    plt.errorbar(range(polys),beta,yerr=(beta+2*v_j),fmt="none",elinewidth=1,capsize=3)
    plt.plot(range(polys),beta,'o')
    plt.legend(["Betas","Confidence interval"])
    plt.title("Confidence interval")
    plt.xlabel("Beta")
    plt.ylabel("Coefficient")
    plt.show()

def surfplot(x,y,z):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x, y, z, cmap=cm.binary,
                           linewidth=0, antialiased=False,alpha=1)

    plt.title("Real terrain data")
    plt.xlabel("X")
    plt.ylabel("Y")
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(30, 120)
    plt.show()

def lambda_MSE_plot(lambdas,MSE):
    fig = plt.figure(figsize=(9,4))
    plt.plot(lambdas,MSE)
    plt.xlabel("Lambda")
    plt.ylabel("MSE")
    plt.show()

def showimage(xshape,yshape,z):
    plt.figure()
    plt.imshow(z.reshape(xshape,yshape), cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def OLS(z,H,X):
    m = len(z.flatten())
    beta_OLS = np.dot(np.linalg.inv(H),np.dot(np.transpose(X),z.flatten()))
    z_fit = np.dot(X,beta_OLS)
    MSE = np.sum((z.flatten()-z_fit.flatten())**2)/m
    R_2 = 1-(np.sum((z.flatten()-z_fit.flatten())**2))/(np.sum((z.flatten()-np.sum(z.flatten())/m)**2))
    var_OLS = np.mean((z_fit.flatten()-np.mean(z_fit.flatten()))**2)
    bias_OLS = np.mean((z.flatten()-np.mean(z_fit.flatten()))**2)
    #extra_term for comparing MSE = bias + variance + extra term
    extra_term = (2./m)*np.sum((z.flatten()-np.mean(z_fit.flatten()))*(np.mean(z_fit.flatten())-z_fit.flatten()))
    print "MSE OLS = ",MSE
    print "R2 OLS = ",R_2
    print "Bias OLS ",bias_OLS
    print "var OLS", var_OLS
    #print var_OLS+bias_OLS+extra_term
    return z_fit,beta_OLS,MSE,R_2,bias_OLS,var_OLS

def Ridge(z,H,X,lambd,polys):
    m = len(z.flatten())
    beta_Ridge = np.dot(np.linalg.inv(H+lambd*np.identity(polys)),np.dot(np.transpose(X),z.flatten()))
    z_fit_ridge = np.dot(X,beta_Ridge)
    MSE_ridge = np.sum((z.flatten()-z_fit_ridge.flatten())**2)/m
    R_2_ridge = 1-(np.sum((z.flatten()-z_fit_ridge.flatten())**2))/(np.sum((z.flatten()-np.sum(z.flatten())/m)**2))
    bias_ridge = np.mean((z.flatten()-np.mean(z_fit_ridge.flatten()))**2)
    var_ridge = np.mean((z_fit_ridge.flatten()-np.mean(z_fit_ridge.flatten()))**2)
    extra_term = (2./m)*np.sum((z.flatten()-np.mean(z_fit_ridge.flatten()))*(np.mean(z_fit_ridge.flatten())-z_fit_ridge.flatten()))
    print "MSE Ridge = ",MSE_ridge
    print "R2 Ridge = ",R_2_ridge
    print "bias_ridge = ",bias_ridge
    print "var_ridge = ",var_ridge
    #print var_ridge+bias_ridge+extra_term
    return z_fit_ridge,beta_Ridge,MSE_ridge,R_2_ridge,bias_ridge,var_ridge

def Lasso(z,H,X,lambd):
    m = len(z.flatten())
    lasso = linear_model.Lasso(alpha=lambd,fit_intercept=True,max_iter=int(10**5))
    lasso.fit(X,z.flatten())
    lasso.coef_[0] = lasso.intercept_
    beta_Lasso = lasso.coef_
    z_fit_lasso = np.dot(X,beta_Lasso)
    MSE_lasso = np.sum((z.flatten()-z_fit_lasso.flatten())**2)/m
    R_2_lasso = 1-(np.sum((z.flatten()-z_fit_lasso.flatten())**2))/(np.sum((z.flatten()-np.sum(z.flatten())/m)**2))
    bias_lasso = np.mean((z.flatten()-np.mean(z_fit_lasso.flatten()))**2)
    var_lasso = np.mean((z_fit_lasso.flatten()-np.mean(z_fit_lasso.flatten()))**2)
    extra_term = (2./m)*np.sum((z.flatten()-np.mean(z_fit_lasso.flatten()))*(np.mean(z_fit_lasso.flatten())-z_fit_lasso.flatten()))
    #print "Lambda = ",lambd
    #print "MSE Lasso = ",MSE_lasso
    #print "R2 Lasso = ",R_2_lasso
    #print "bias lasso = ",bias_lasso
    #print "var lasso = ",var_lasso
    #print var_lasso+bias_lasso+extra_term
    return z_fit_lasso,beta_Lasso,MSE_lasso,R_2_lasso,bias_lasso,var_lasso

def kfold(x,y,z,k,type,lambd,p):
    m = int(len(x.flatten()))
    X = designmatrix(x.flatten(),y.flatten(),p)
    polys = sum(range(1,p+2)) #Number of polynomials
    z2 = np.zeros(shape=(len(x.flatten())/k,k)) #array of test data
    z2_fit = np.zeros(shape=(len(x.flatten())/k,k)) #Fitted test data
    for i in range(len(x.flatten())/k):
        #Training set with x,y,z[everything up to i*k and everything after k+i*k]
        x_train = np.concatenate((x.flatten()[:(i*k)],x.flatten()[(k+i*k):]))
        y_train = np.concatenate((y.flatten()[:(i*k)],y.flatten()[(k+i*k):]))
        z_train = np.concatenate((z.flatten()[:(i*k)],z.flatten()[(k+i*k):]))
        #Testing set with x,y,z[everything between i*k and k+i*k]
        x_test = x.flatten()[(i*k):(k+i*k)]
        y_test = y.flatten()[(i*k):(k+i*k)]
        z_test = z.flatten()[(i*k):(k+i*k)]
        X_train = designmatrix(x_train,y_train,p)
        X_test = designmatrix(x_test,y_test,p)
        H = np.dot(np.transpose(X_train),X_train) #To be used in finding betas
        if(type=="Ridge"):
            beta = np.dot(np.linalg.inv(H+lambd*np.identity(polys)),np.dot(np.transpose(X_train),z_train.flatten()))
        elif(type=="Lasso"):
            lasso = linear_model.Lasso(alpha=lambd,fit_intercept = True,max_iter=int(10**6),tol=0.0001,precompute=True)
            lasso.fit(X_train,z_train.flatten())
            lasso.coef_[0] = lasso.intercept_
            beta = lasso.coef_

        else:
            beta = np.dot(np.linalg.inv(H),np.dot(np.transpose(X_train),z_train.flatten()))
        z2_fit[i] = np.dot(X_test,beta) #+ np.mean(z) #Adding mean for centered data
        z2[i] = z_test #+ np.mean(z) ##Adding mean for centered data
        #Terminal progress bar
        sys.stdout.write("\r kfold progress: %.2f%%"%(float(i+1)*100/(len(x.flatten())/k)))
        sys.stdout.flush()

    bias_kfold = np.mean((z2.flatten()-np.mean(z2_fit.flatten()))**2)
    MSE_k_fold = np.mean((z2.flatten()-z2_fit.flatten())**2)
    R2_k_fold = 1-(np.sum((z2.flatten()-z2_fit.flatten())**2))/(np.sum((z2.flatten()-np.sum(z2.flatten())/(m))**2))
    var_kfold = np.mean((z2_fit.flatten()-np.mean(z2_fit.flatten()))**2)
    #e_term (extra term) for comparing MSE = bias + variance + extra term
    e_term = (2./m)*np.sum((z2.flatten()-np.mean(z2_fit.flatten()))*(np.mean(z2_fit.flatten())-z2_fit.flatten()))

    print "-------------------------"
    print "MSE_kfold %s = "%type,MSE_k_fold
    print "R2_kfold %s ="%type,R2_k_fold
    print "Bias kfold ", bias_kfold
    print "var k-fold ", var_kfold
    #print "------------------------"
    return MSE_k_fold,R2_k_fold
