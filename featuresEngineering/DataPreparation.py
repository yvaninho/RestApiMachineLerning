#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:38:10 2019

@author: jeff
"""


import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import sys
import itertools
import matplotlib.pyplot as plt
import abc
import spark
   
#######################################################################
# module contenant les fonctions permettant 
######################################################################    
def readTable(name):
    max_date = spark.sql("select max(ref_chargement) from" + "name")
    ref_charg = max_date.collect()[0][0]

    table_add_0= spark.sql("select * from "+"name"+ "where ref_chargement='{}'".format(ref_charg))
    table_add_0 = table_add_0.toPandas()
    typologie_add_0= spark.sql("select * from prd_tech.b_r_datasmart_ml_02_typologie_03_predict_primitive_all where ref_chargement='{}'".format(ref_charg))
    typologie_add_0 = typologie_add_0.toPandas()
    
 
    
    

def test(x):
    mot='encombrement'
    if mot in x :
        return True
    else:
        return False

def longest_common_substring(x):
    x=x.lower()
    strs=x.split(',')
    s1=strs[0]
    s2=strs[1]
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]
   

def createData(typologie_add_0,table_add_0 ):

    typologie_add_0.drop(['typologie2','confiance2','confiance1'],axis=1,inplace=True)
    typologie_add_0['nom_champ1']=typologie_add_0['nom_champ']
    typologie_add_0['nom_champ2']=typologie_add_0['nom_champ']
    typologie_add_0['typologie2']=typologie_add_0['typologie1']

    table_add_1=table_add_0.loc[:,['nom_table','nom_champ']].merge(typologie_add_0.loc[:,['typologie1','nom_champ']],on=['nom_champ'])
    table_add_lib=table_add_1.loc[(table_add_1['typologie1']=='libelle'),['nom_table','nom_champ']]
    table_add_cod=table_add_1.loc[(table_add_1['typologie1']=='code/identifiant'),['nom_table','nom_champ']]

    table_add_lib['nom_champ2']=table_add_lib['nom_champ']
    table_add_lib.drop(['nom_champ'],axis=1,inplace=True)

    table_add_cod['nom_champ1']=table_add_cod['nom_champ']
    table_add_cod.drop(['nom_champ'],axis=1,inplace=True)


    candidat_0 = table_add_cod.merge(table_add_lib,on=['nom_table'])
    candidat_1 = candidat_0.drop(['nom_table'],axis=1)


    candidat_1['temp']=candidat_1['nom_champ1']+','+candidat_1['nom_champ2']
    A=np.unique(candidat_1['temp'],return_index=True)
    candidat_2=candidat_1.iloc[A[1],:].drop(['temp'],axis=1)
    candidat_2['source']='dicoADD'
    del candidat_1,candidat_0
    candidat_2['temp']=candidat_2['nom_champ1']+','+candidat_2['nom_champ2']
    candidat_2['longuest']=candidat_2['temp'].apply(longest_common_substring)
    candidat_2['len_longuest']=candidat_2['longuest'].apply(len)
    candidat_2.drop(['temp','longuest'],axis=1,inplace=True)

    candidat_2.sort_values(by=['nom_champ1','len_longuest'],ascending=[True,False],inplace=True)
    candidat_2.reset_index(drop=True,inplace=True)

    candidat_2_tmp=candidat_2.copy()
    A=np.unique(candidat_2_tmp['nom_champ1'],return_index=True)
    candidat_3=candidat_2_tmp.iloc[A[1],:].reset_index(drop=True)
    candidat_2_tmp=candidat_2_tmp.drop(A[1]).reset_index(drop=True)
    for i in range(9):
        A=np.unique(candidat_2_tmp['nom_champ1'],return_index=True)
        candidat_3=pd.concat([candidat_3,candidat_2_tmp.iloc[A[1],:]],ignore_index=True).reset_index(drop=True)
        candidat_2_tmp=candidat_2_tmp.drop(A[1]).reset_index(drop=True)
    
    res_test1=candidat_2['nom_champ1'].apply(test)
    res_test2=candidat_2['nom_champ2'].apply(test)


    res_test1=candidat_3['nom_champ1'].apply(test)
    res_test2=candidat_3['nom_champ2'].apply(test)
    data = [res_test1,res_test2]
    return data 


def datasetTrain_add(name):
    max_date = spark.sql("select max(ref_chargement) from" + "name")
    ref_charg = max_date.collect()[0][0]
    
    dataset_code_lib_add_ok0=spark.sql("select * from prd_tech.b_r_datasmart_ml_03_code_libelle_03_learn_add where ref_chargement='{}'".format(ref_charg))
    dataset_code_lib_add_ok0 = dataset_code_lib_add_ok0.toPandas()

    dataset_code_lib_add_ok1=dataset_code_lib_add_ok0[pd.isnull(dataset_code_lib_add_ok0['statut'])==False]
    dataset_code_lib_add_ok1['temp']=dataset_code_lib_add_ok1['nom_champ1']+','+dataset_code_lib_add_ok1['nom_champ2']
    dataset_code_lib_add_ok1.reset_index(drop=True,inplace=True)
    A=np.unique(dataset_code_lib_add_ok1['temp'],return_index=True)
    dataset_code_lib_add_ok2=dataset_code_lib_add_ok1.loc[A[1],:].reset_index(drop=True)
    dataset_code_lib_add_ok2.drop(['temp'],axis=1,inplace=True)
    dataset_code_lib_add_ok2['source']='manuelADD'

    dataset_code_lib_add_ok2['temp']=dataset_code_lib_add_ok2['nom_champ1']+','+dataset_code_lib_add_ok2['nom_champ2']
    dataset_code_lib_add_ok2['longuest']=dataset_code_lib_add_ok2['temp'].apply(longest_common_substring)
    dataset_code_lib_add_ok2['len_longuest']=dataset_code_lib_add_ok2['longuest'].apply(len)
    dataset_code_lib_add_ok2.drop(['temp','longuest'],axis=1,inplace=True)
    
    return dataset_code_lib_add_ok2

    del dataset_code_lib_add_ok0,dataset_code_lib_add_ok1       

def datasetTrain_ds(typologie_add_0):
    
    max_date = spark.sql("select max(ref_chargement) from" + "name")
    ref_charg = max_date.collect()[0][0]
 
    dataset_code_lib_DS_ok0=spark.sql("select * from prd_tech.b_r_datasmart_ml_03_code_libelle_03_learn_ds where ref_chargement='{}'".format(ref_charg))
    dataset_code_lib_DS_ok1 = dataset_code_lib_DS_ok0.toPandas()

    dataset_code_lib_DS_ok1['temp']=dataset_code_lib_DS_ok1['nom_champ1']+','+dataset_code_lib_DS_ok1['nom_champ2']
    A=np.unique(dataset_code_lib_DS_ok1['temp'],return_index=True)
    dataset_code_lib_DS_ok2=dataset_code_lib_DS_ok1.loc[A[1],:].reset_index(drop=True)
    dataset_code_lib_DS_ok2.drop(['temp'],axis=1,inplace=True)
    dataset_code_lib_DS_ok3=dataset_code_lib_DS_ok2[pd.isnull(dataset_code_lib_DS_ok2['statut'])==False]
    dataset_code_lib_DS_ok4=dataset_code_lib_DS_ok3.merge(typologie_add_0.loc[:,['nom_champ1','typologie1']],on=['nom_champ1'])
    dataset_code_lib_DS_ok5=dataset_code_lib_DS_ok4.merge(typologie_add_0.loc[:,['nom_champ2','typologie2']],on=['nom_champ2'])
    dataset_code_lib_DS_ok6=dataset_code_lib_DS_ok5[dataset_code_lib_DS_ok5['typologie1']=='code/identifiant']
    dataset_code_lib_DS_ok7=dataset_code_lib_DS_ok6[dataset_code_lib_DS_ok6['typologie2']=='libelle']
    dataset_code_lib_DS_ok7['source']='manuelDS'
    
    dataset_code_lib_DS_ok7['temp']=dataset_code_lib_DS_ok7['nom_champ1']+','+dataset_code_lib_DS_ok7['nom_champ2']
    dataset_code_lib_DS_ok7['longuest']=dataset_code_lib_DS_ok7['temp'].apply(longest_common_substring)
    dataset_code_lib_DS_ok7['len_longuest']=dataset_code_lib_DS_ok7['longuest'].apply(len)
    dataset_code_lib_DS_ok7.drop(['temp','longuest'],axis=1,inplace=True)
    
    del dataset_code_lib_DS_ok0,dataset_code_lib_DS_ok1,dataset_code_lib_DS_ok2,dataset_code_lib_DS_ok3,dataset_code_lib_DS_ok4
    del dataset_code_lib_DS_ok5,dataset_code_lib_DS_ok6
    return dataset_code_lib_DS_ok7  


def diff_str(x):
    x=x.lower()
    strs=x.split(',')
    str1=strs[0]
    str2=strs[1]
    charac=strs[2]
    return (str2.count(charac)-str1.count(charac))

       
def buildDataset(dataset_code_lib_DS_ok7, dataset_code_lib_add_ok2,candidat_2):
    dataset_all_0=pd.concat([dataset_code_lib_add_ok2,dataset_code_lib_DS_ok7,candidat_2],ignore_index=True)
    dataset_all_1=dataset_all_0.copy()
    dataset_all_1['temp']=dataset_all_1['nom_champ1']+','+dataset_all_1['nom_champ2']
    dataset_all_1.reset_index(drop=True,inplace=True)
    A=np.unique(dataset_all_1['temp'],return_index=True)
    dataset_all_2=dataset_all_1.loc[A[1],dataset_code_lib_DS_ok7.columns]
    dataset_all_2.reset_index(drop=True,inplace=True)
    
    dataset_all_3=dataset_all_2.copy()
    dataset_all_3['diff_len']=dataset_all_3['nom_champ1'].apply(len)-dataset_all_3['nom_champ2'].apply(len)
    dataset_all_3['mean_len']=(dataset_all_3['nom_champ1'].apply(len)+dataset_all_3['nom_champ2'].apply(len))/2
    tmp=np.concatenate([np.array(dataset_all_3['nom_champ1'].apply(len),ndmin=2),np.array(dataset_all_3['nom_champ2'].apply(len),ndmin=2)])
    dataset_all_3['min_len']=np.min(tmp,axis=0)
    dataset_all_3['ratio']=np.abs(dataset_all_3['diff_len'])*1.0/dataset_all_3['mean_len']
    dataset_all_3.drop(['diff_len'],axis=1,inplace=True)
    dataset_all_4=dataset_all_3.copy()
    
    dataset_all_4['ratio_common_min_len']=dataset_all_4['len_longuest']/dataset_all_4['min_len']

    for i in range(26):
        dataset_all_4['temp']=dataset_all_4['nom_champ1']+','+dataset_all_4['nom_champ2']+','+chr(97+i)
        dataset_all_4[chr(97+i)]= dataset_all_4['temp'].apply(diff_str)
        dataset_all_4.drop(['temp'],axis=1,inplace=True)
    for i in range(10):
        dataset_all_4['temp']=dataset_all_4['nom_champ1']+','+dataset_all_4['nom_champ2']+','+chr(48+i)
        dataset_all_4[chr(48+i)]= dataset_all_4['temp'].apply(diff_str)
        dataset_all_4.drop(['temp'],axis=1,inplace=True)
    array_chiff=np.array(dataset_all_4.loc[:,['0','1','2','3','4','5','6','7','8','9']])
    array_chiff_p=array_chiff.copy()
    array_chiff_n=array_chiff.copy()
    array_chiff_p[array_chiff_p<0]=0
    array_chiff_n[array_chiff_p>0]=0
    dataset_all_4['chiff_p']=np.sum(array_chiff_p,axis=1)
    dataset_all_4['chiff_n']=np.sum(array_chiff_n,axis=1)
    dataset_all_4.drop(['0','1','2','3','4','5','6','7','8','9'],axis=1,inplace=True)
    dataset_all_4.reset_index(drop=True,inplace=True)
    return 

def finalDatasetCL_add(dataset_all_4):
    listDataOutput = []
    noms_tmp=dataset_all_4.loc[:,['nom_champ1','nom_champ2']]
    X_tmp=dataset_all_4.iloc[:,range(-33,0,1)]
    y_tmp=dataset_all_4['statut']

    filtre_training=dataset_all_4['source']!='dicoADD'
    filtre_output=dataset_all_4['source']!='manuelDS'
    
    X=X_tmp[filtre_training]
    y=y_tmp[filtre_training]
    noms=noms_tmp[filtre_training]
    X.reset_index(drop=True,inplace=True)
    y.reset_index(drop=True,inplace=True)
    noms.reset_index(drop=True,inplace=True)

    X_all=X_tmp[filtre_output]
    y_all=y_tmp[filtre_output]
    noms_all=noms_tmp[filtre_output]
    X_all.reset_index(drop=True,inplace=True)
    y_all.reset_index(drop=True,inplace=True)
    noms_all.reset_index(drop=True,inplace=True)
    listDataOutput = [X,y,X_all,y_all, noms_all]
    return listDataOutput
    
