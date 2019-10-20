# coding: utf-8
import os
import sys
import numpy as np
import pandas as pd


class MultiClassBernouilliNB():
    
    def __init__(self, alpha): 
        self.alpha = alpha
        self.X     = None
        self.y     = None
        
        
    def _compute_marginal_probabilities(self):
        """ 
        Marginal probabilities for each class
        """
        self.marginals = np.empty(len(self.classes))
        for k in self.classes:
            Y_k = self.data[self.data[:,-1] == k]
            self.marginals[k] = float(Y_k.shape[0])/float(self.data.shape[0])
        
        
    def _compute_priors(self):
        """
        Compute the priors matrix P(Xj|Yk)
        """
        self.priors = np.empty((self.n_features,len(self.classes),))
        
        for k in self.classes:
            Y_k = self.data[self.data[:,-1] == k]
            
            for j in range(self.n_features):    
                # number of times xj=1 and y=k
                Xj1_Yk   = Y_k[Y_k[:,j] == 1.0]
                
                # Add Laplace smoothing (parameter self.alpha)
                numerator   = self.alpha + Xj1_Yk.shape[0]
                denominator = self.alpha*len(self.classes) + Y_k.shape[0]
                theta_jk    = float(numerator)/float(denominator)
                
                self.priors[j,k] = theta_jk
        
    
    def fit(self, X, y):
        """
        For each class in y, compute the marginal probabilities P(Y=k)
        For each feature Xj, compute the contional pronability P(Xj|Y)
        """
        # Train set X and y
        self.X = X
        self.y = y
        self.data = np.c_[self.X, self.y]
        
        # number of classes to predict
        self.classes = set(self.y)
        # number of features 
        self.n_features = X.shape[1]
        
        # compute the marginal P(Y=k) and prior probabilities P(Xj|Y)
        self._compute_marginal_probabilities()
        self._compute_priors()
        
        # log both the marginal and prior probabilities for numerical stability
        self.log_marginals   = np.log(self.marginals)
        self.log_priors      = np.log(self.priors)
        self.log_1minusprior = np.log(1.0 - self.priors)
        
        
    def predict(self, X_test):
        """
        Apply equation from lecture 10, slide 6 (matrix version),
        to pick the most likely class
        """
        self.X_test = X_test
        
        # Predictions proportional to likelihood for class 0 and class 1
        self.likelihoods = np.empty((X_test.shape[0],len(self.classes),))
        self.pred_class  = np.empty((X_test.shape[0],1,))
        
        for i in range(self.X_test.shape[0]):
            # individual x to classify
            x = X_test[i,:]
            
            for k in self.classes:
                pred_yk  = self.log_marginals[k]
                pred_yk += np.dot(x, self.log_priors[:,k])
                pred_yk += np.dot((1.0-x), self.log_1minusprior[:,k])
                
                self.likelihoods[i,k] = pred_yk
                
            # predicted class corresponds to the maximum value of pred_yk for this instance
            self.pred_class[i,0] = np.where(self.likelihoods[i,:] == np.amax(self.likelihoods[i,:]))[0][0]
        return self.pred_class
        
        

