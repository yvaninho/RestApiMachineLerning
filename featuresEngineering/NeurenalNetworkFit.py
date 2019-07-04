# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:13:13 2019

@author: b011sjh
"""

class NeuronalNetworkFit:
    
    
    
    def __init__(Xn,Y):
        
        self.__Xn = Xn
        self.__Y = Y
        
    
    def neuronalNetworkFit(X_n, Y):
        
        
        y_test_concat=[]
        y_pred_concat=[]
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
        modelFit = model.fit(X_Normalise, y, epochs=nb_epochs, batch_size=batch_size)
        return modelFit