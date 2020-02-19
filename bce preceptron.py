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
        return np.c_[scaled_features, np.ones([scaled_features.shape[0],1])]
    
    def Activation(self, hypothesis):
        activ=(1/(1+np.exp(-(hypothesis))))
        #print(activ)
        #return 1 if activ>=THRESHOLD else 0
        return activ
    def NewActivation(self,data):
        return 1 if data>=THRESHOLD else 0
        
        
    def Hypothesis (self, scaled_features,weights):
        hypo=np.dot(scaled_features,weights)
        output=self.Activation(hypo)
#        return 1 if output>=THRESHOLD else 0
        return output
    
    def Loss_Func(self,prediction, predictor):
        
        #print(l)
        l1=np.log(np.subtract(1,prediction))
        l2=np.subtract(1,predictor)
        l3=np.asarray(np.dot(l1,l2))
        l4=np.log(prediction)
        l5=np.asarray(np.dot(predictor,l4))
        add=np.array([l3,l5])
        l6=np.sum(add)

        return l6
    
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
                    
                    Newoutput=self.NewActivation(output)
                    out_hist.append(Newoutput)
                    
                    loss_iter+=self.Loss_Func(output,y)
                    self.weights[0]=self.weights[0]-(lr*(Newoutput-y)*x[0])
#                    print(self.weights[0])
                    self.weights[1]=self.weights[1]-(lr*(Newoutput-y)*x[1])
#                    print(self.weights[1])
                    self.weights[2]=self.weights[2]-(lr*(Newoutput-y)*x[2])
                    self.weights[3]=self.weights[3]-(lr*(Newoutput-y)*x[3])
                    self.weights[4]=self.weights[4]-(lr*(Newoutput-y)*x[4])
                    self.weights[5]=self.weights[5]-(lr*(Newoutput-y)*x[5])
                    self.weights[6]=self.weights[6]-(lr*(Newoutput-y)*x[6])
                loss_hist.append(loss_iter)
                loss_iter=0
                accuracy.append(accuracy_score(out_hist,self.predictor))
#                print(accuracy[i])
                if(accuracy[i]>max_accuracy):
                    max_accuracy=accuracy[i]
                    
                    print(max_accuracy)
                    chkptw=self.weights
                out_hist.clear()
            print(max_accuracy)
            return [chkptw,max_accuracy]

data=pd.read_csv("C:/Users/ujjwal/Desktop/_/ml/mobile_cleaned-1549119762886.csv")
data_thin=data[['aperture','battery_capacity','brand_rank','stand_by_time','screen_size','price','video_resolution']]
data_thin.head
predictor_target=data[['is_liked']].values

perc=PERCEPTRON(data_thin,predictor_target)
final=perc.Fit(5000,0.24)