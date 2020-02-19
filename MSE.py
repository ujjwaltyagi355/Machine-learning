# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 19:08:49 2020

@author: ujjwal
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
THRESHOLD=0.2

class PERCEPTRON:
    def __init__(self, features, predictor):
        self.features=features
        self.predictor=predictor
    
    def Scaling(self, features):
        scaled_features=StandardScaler().fit_transform(features)
        #print(np.c_[scaled_features, np.ones([scaled_features.shape[0],1])])
        return np.c_[scaled_features, np.ones([scaled_features.shape[0],1])]
    
    def Activation(self, hypothesis):
        activ=(1/(1+np.exp(-(hypothesis))))
        #print(activ)
        return 1 if activ>=THRESHOLD else 0
        
    def Hypothesis (self, scaled_features,weights):
        hypo=np.dot(scaled_features,weights)
        output=self.Activation(hypo)
        return output
    
    def Loss_Func(self,prediction, predictor):
        loss= np.square((prediction-predictor))
        #print(loss[0])
        return loss[0]
    
    def Fit(self,epochs,lr):
        self.scaled_features=self.Scaling(self.features)
        out_hist=[]
        loss_hist=[]
        accuracy=[]
        max_accuracy=0
        loss_iter=100
        self.weights=np.ones([self.scaled_features.shape[1]])
        #print(self.weights.shape)
        while (loss_iter>=1):
            for i in range(epochs):
                for x,y in zip(self.scaled_features,self.predictor):
                    
                    output=self.Hypothesis(x, self.weights)
                    #print(output)
                    out_hist.append(output)
                    #print(out_hist)
                    loss_iter+=self.Loss_Func(output,y)
                    #print(loss_iter)
                    self.weights[0]=self.weights[0]-(lr*(output-y)*x[0])
                    self.weights[1]=self.weights[1]-(lr*(output-y)*x[1])
                    self.weights[2]=self.weights[2]-(lr*(output-y)*x[2])
                    self.weights[3]=self.weights[3]-(lr*(output-y)*x[3])
                    self.weights[4]=self.weights[4]-(lr*(output-y)*x[4])
                    self.weights[5]=self.weights[5]-(lr*(output-y)*x[5])
                    self.weights[6]=self.weights[6]-(lr*(output-y)*x[6])
                loss_hist.append(loss_iter)
                #print(loss_iter)
                loss_iter=0
                accuracy.append(accuracy_score(out_hist,self.predictor))
                if(accuracy[i]>max_accuracy):
                    max_accuracy=accuracy[i]
                    chkptw=self.weights
                out_hist.clear()
            plt.plot(loss_hist)
            plt.show()
    
            print(max_accuracy)
        
            
            return [chkptw,max_accuracy]
       
data=pd.read_csv("C:/Users/ujjwal/Desktop/_/ml/mobile_cleaned-1549119762886.csv")
data_thin=data[['aperture','battery_capacity','brand_rank','stand_by_time','screen_size','price','video_resolution']]
data_thin.head
#print(data_thin)
predictor_target=data[['is_liked']].values
#print(predictor_target)

perc=PERCEPTRON(data_thin,predictor_target)
final=perc.Fit(5000,0.24)