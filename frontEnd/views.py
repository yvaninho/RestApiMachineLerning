#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:51:52 2019

@author: jeff
"""
from featuresEngineering import NeuronalNetworkCodeLibelle as ncl
from flask import Flask, render_template, jsonify
import pandas as pd
import redis

######################configuration de redis #########################
#

#
#################################################################

app = Flask(__name__)
app.config.from_object('config')

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')
data = pd.read_excel('/home/jeff/Documents/workspace/Machine_learning_semantique/predColdeLibelle2.xlsx')
data1 = data.to_html( classes= 'table table-striped')

fichier = open("tablehtml.html", "w")
fichier.write(data1)
fichier.close()




@app.route('/')
@app.route('/index/')
def index():
    return render_template('index.html')

@app.route('/result/')
def result():
    data = pd.read_excel('/home/jeff/Documents/workspace/Machine_learning_semantique/predColdeLibelle.xlsx')
    return render_template('result.html', data = data )


@app.route('/api/codelibelles/')
def dashboard():
    return df.to_json(orient='split')

def readData():
    data = pd.read_csv('/home/jeff/Documents/workspace/Machine_learning_semantique/featuresEngineering/mydatatest.csv',header=None , delim_whitespace=True)
    #data.set_index(['nom_table'], inplace=True)
    #data.index.name=None
    return data

@app.route('/train_code_libelle/')
def letrainCodeLibelle():
	queue = redis.StrictRedis(
			host ='127.0.0.1',
			port = 6379)
	channel  = queue.pubsub()
	queue.publish('trainChannel','start')
    #listData = ncl.TrainPredictCodeLibelle.readData()
    ##print("ok")
    #X = listData[0]
    #y = listData[1]
    ##print(type(X))
    ##print(type(y))
    ##data = classTrain.readData()
    #resultTrain = ncl.TrainPredictCodeLibelle().trainCodeLibelle(X,y)
    #p#rint(len(resultTrain))
    ##print(type(resultTrain[1]))
    #ctrainCodeLibelle(X,y)
    #resultat = pd.DataFrame(resultTrain[1])
    #resultat1 = pd.DataFrame(resultTrain[0])
    #resultat.to_excel()
    ##print(type(resultat))
	return render_template('train.html', data1 = data1)


@app.route('/train/')
def lestrains():
    #slistData = ncl.TrainPredictCodeLibelle.readData()

    prediction = ncl.TrainPredictCodeLibelle().predictCodeLibelle()
    return render_template('train.html', prediction=prediction)
    #print(prediction)

if __name__ == "__main__":
    app.debug =True
    app.run()
