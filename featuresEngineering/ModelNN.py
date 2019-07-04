# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:21:20 2019

@author: b011sjh
"""
from abc import ABC 
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import sys
import itertools
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)



class ModelNN(ABC):
    
    

    
    def __init__ (self, nb_epochs =25, nb_hidden_layers=2,neurons=100,optimizer_str='sgd lr=0.01, decay=1e-6, momentum=0.9, nesterov=True',batch_size=64):
        self.__nb_epochs = nb_epochs
        self.__nb_hidden_layers = nb_hidden_layers
        self.__neurons = neurons
        self.__optimizer_str = optimizer_str
        self.__batch_size = batch_size
     
 """
     construction des setters et getters



"""       
    @setter
    def setNbEpochs(self, nb_epochs ):
        self.__nb_epochs = nb_epochs
    
    
    @setter
    def setNbHiddenLayers(self, nb_hidden_layers ):
        self.__nb_hidden_layers = nb_hidden_layers   
        

    @setter
    def setNeurons(self, neurons ):
        self.__neurons = neurons     
        
    @setter
    def setNeurons(self, optimizer_str ):
        self.__optimizer_str = optimizer_str  
    
    
    @abstractmethod
    def numToClass(self):
        
        
        
    @abstractmethod
    def plot_confusion_matrice(self):
        
        
    
        
    @abstractmethod
    def neuronalNetworkFit(self,X, Y):
        

        y_test_concat=[]
        y_pred_concat=[]
        for idx_train , idx_test in skf.split(X,y)
            normalizX=preprocessing.StandardScaler()
            X_Normalise = normalizX.fit_transform(X)

            model = Sequential()
            model.add(Dense(units=neurons, activation='relu', input_dim=X.shape[1]))
            model.add(Dropout(0.5))
            if nb_hidden_layers>0:
                model.add(Dense(units=neurons, activation='relu', input_dim=neurons))
                model.add(Dropout(0.5))
            if nb_hidden_layers>1:
                model.add(Dense(units=neurons, activation='relu', input_dim=neurons))
                model.add(Dropout(0.5))
            if nb_hidden_layers>2:
                model.add(Dense(units=neurons, activation='relu', input_dim=neurons))
            model.add(Dense(units=1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
            #modelFit = model.fit(X_Normalise, y, epochs=nb_epochs, batch_size=batch_size)
            for i in range(nb_epochs):
                print(i+1,"/",nb_epochs)
                res=model.fit(X_train, Y_train, epochs=1, batch_size=batch_size)
                score_train=res.history['acc'][0]
                score_test=model.evaluate(X_test,Y_test)
                train_accuracy_all[compteur,i]=score_train
                test_accuracy_all[compteur,i]=score_test[1]
            compteur+=1
    
            y_test_class=pd.DataFrame(Y_test)['statut'].apply(numToClass)
            y_test_concat.append(y_test_class)
            y_pred_concat.append(model.predict(X_test, batch_size=128))
            
        return modelFit
        
    
    