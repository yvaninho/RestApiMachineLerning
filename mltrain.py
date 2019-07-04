#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from featuresEngineering import NeuronalNetworkCodeLibelle as ncl
import pandas as pd
import redis
from redis import StrictRedis
import pdb
################## redis configuration ###################
r = redis.StrictRedis(host = 'localhost', port= 6379)
p = r.pubsub()

p.subscribe('trainChannel')

#message = p.listen()
#data = str(message.get("data"))
#pdb.set_trace()
#for chnel in p.listen():
while True:
	message = p.get_message()
	#data = str(chnel.get("data"))
	if message:
		print("subscriber :%s" % message['data'] )
		listData = ncl.TrainPredictCodeLibelle.readData()
		"""X = listData[0]
		y = listData[1]
		resultTrain = ncl.ncl.TrainPredictCodeLibelle().trainCodeLibelle(X,y)
		ctrainCodeLibelle(X,y)
		resultatresultat = pd.DataFrame(resultTrain[1])
		resultat1 = pd.DataFrame(resultTrain[0])
		resultat.to_excel("resultat_code_libelle2.xlsx")"""
		prediction = ncl.TrainPredictCodeLibelle().predictCodeLibelle()
