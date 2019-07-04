#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:31:26 2019

@author: jeff
"""

from featuresEngineering import NeuronalNetworkCodeLibelle as ncl
#from flask import Flask, render_template, jsonify
import pandas as pd




    

if __name__ == "__main__":
    # execute only if run as a script
    """data = pd.read_csv('/home/jeff/Documents/workspace/Machine_learning_semantique/featuresEngineering/mydatatest.csv' ,header=0, delim_whitespace=True)
    noms_all=data.loc[:,['nom_table','nom_champ1','nom_champ2']]
    filtre=(pd.isnull(data['statut'])==False)
    X_all=data.drop(['nom_table','nom_champ1','nom_champ2','statut',],axis=1)# on delecte les variable non explicative
    X=X_all[filtre]## on prend les observations dont le status est connu
    y=data.loc[filtre,['statut']] # labels 
    X.reset_index(drop=True,inplace=True)
    y.reset_index(drop=True,inplace=True)"""
    listData = ncl.TrainPredictCodeLibelle.readData()
    print("ok")
    ##X = listData[0]
    ##y = listData[1]
    ##X_all = listData[2]
    ##print(type(X))
    ##print(type(y))
    #data = classTrain.readData()
    ##resultTrain = ncl.TrainPredictCodeLibelle().trainCodeLibelle(X,y)
    ##print(len(resultTrain))
    ##print(type(resultTrain[1]))
    #trainCodeLibelle(X,y)
    ##resultat = pd.DataFrame(resultTrain[1])
    ##resultat.to_excel("resultat_code_libelle.xlsx")
    ##print(type(X_all))
    prediction = ncl.TrainPredictCodeLibelle().predictCodeLibelle()
    print(prediction)
    