# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:51:28 2019

@author: b011sjh
"""



from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn import preprocessing
import pandas as pd
import numpy as np


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import sys
import itertools
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import logging
logger = logging.getLogger(__name__)
from sklearn.externals import joblib



class TrainPredictCodeLibelle():

    nb_epochs=25 #
    nb_hidden_layers=2 # nombre de couches cachés
    neurons=100 # Nombre de neurone
    optimizer='sgd' #adam
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer_str='sgd lr=0.01, decay=1e-6, momentum=0.9, nesterov=True'
    batch_size=64
    n_splits=5
    skf = StratifiedShuffleSplit(n_splits=n_splits,test_size=0.2,random_state=0)
    """ @param X est l'ensemble d'observation qui permettrons de prédire y
    """
    def __init__ (self, nb_epochs =25, nb_hidden_layers=2,neurons=100,optimizer=optimizer,batch_size=64):
        self.nb_epochs = nb_epochs
        self.nb_hidden_layers = nb_hidden_layers
        self.neurons = neurons
        self.optimizer = optimizer
        self.batch_size = batch_size
        #self.__n_splits = n_splits
        """
    def get_NbEpochs(self ):
        return self.__nb_epochs



    def get_NbHiddenLayers(self ):
        return self.__nb_hidden_layers



    def get_Neurons(self ):
        return self.__neurons


    def get_Optimizer_str(self ):
        return self.__optimizer_str

    def get_batch_size(self):
        return self.__batch_size

    #@classmethod
    #def get_Nsplits(self):
        #return self.__n_splits
"""
    #@classmethod
    def numToClass(self,x):
        if x==0:
            return 'non lié'
        else:
            return 'lié'


    def plot_confusion_matrix(self,cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')







    def readData():

        data = pd.read_csv('/home/jeff/Documents/workspace/Machine_learning_semantique/featuresEngineering/mydatatest.csv',header=0, delim_whitespace=True)
        noms_all=data.loc[:,['nom_table','nom_champ1','nom_champ2']]
        filtre=(pd.isnull(data['statut'])==False)
        X_all=data.drop(['nom_table','nom_champ1','nom_champ2','statut',],axis=1)# on delecte les variable non explicative
        X=X_all[filtre]## on prend les observations dont le status est connu
        y=data.loc[filtre,['statut']] # labels
        X.reset_index(drop=True,inplace=True)
        y.reset_index(drop=True,inplace=True)
        L = [X, y ,X_all,noms_all]
        return L

    def saveModel(model):
        joblib.dump(model, 'myModel.pkl')


    def matriceConfusion(self,y_test_concat, y_pred_concat,train_accuracy_all,test_accuracy_all):
        """ Ma fonction matriceConfusion prend une liste des
        statuts test et une liste de statuts predis
        elle permet de construire la matice de confusion




        """
        y_test_concat_all=y_test_concat[0]
        y_pred_concat_all=y_pred_concat[0]
        for i in range(1,self.n_splits):
            y_test_concat_all=np.concatenate([y_test_concat_all,y_test_concat[i]],axis=0)
            y_pred_concat_all=np.concatenate([y_pred_concat_all,y_pred_concat[i]],axis=0)

        seuil=0.5
        y_pred_concat2= (y_pred_concat_all > seuil)*1
        y_test_pred_class=pd.DataFrame(y_pred_concat2,columns=['statut'])['statut'].apply(self.numToClass)

        cnf_matrix= confusion_matrix(y_test_concat_all, y_test_pred_class)

        plt.figure()
        classes=  ['non lié','lié']
        self.plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,title='Test - NN - Normalized confusion matrix')
        plt.show()
        accuracy_model={}
        for i in range(len(classes)):
            classe=classes[i]
            accuracy_model[classe]=cnf_matrix[i,i]/np.sum(cnf_matrix[:,i])


        plt.figure()
        train_accuracy_mean=np.mean(train_accuracy_all,axis=0)
        train_accuracy_std=np.std(train_accuracy_all,axis=0)
        test_accuracy_mean=np.mean(test_accuracy_all,axis=0)
        test_accuracy_std=np.std(test_accuracy_all,axis=0)
        plt.plot(range(len(train_accuracy_mean)),train_accuracy_mean,'r')
        plt.plot(range(len(test_accuracy_mean)),test_accuracy_mean,'b')
        plt.fill_between(range(len(train_accuracy_mean)), train_accuracy_mean-2*train_accuracy_std, train_accuracy_mean+2*train_accuracy_std, color='r', alpha=0.2)
        plt.fill_between(range(len(test_accuracy_mean)), test_accuracy_mean-2*test_accuracy_std, test_accuracy_mean+2*test_accuracy_std, color='b', alpha=0.2)
        plt.plot([0,self.nb_epochs],[0.85,0.85],'k--')
        plt.plot([0,self.nb_epochs],[0.9,0.9],'k--')
        plt.ylim([0.3, 1])
        plt.xlim([0, self.nb_epochs])
        plt.title('hidden_layer:'+str(self.nb_hidden_layers)+'x'+str(self.neurons)+' optim:' + self.optimizer_str +' batch:' +str(self.batch_size))
        plt.legend(['train','test'])
        plt.show()
        return accuracy_model



    def trainCodeLibelle(self, X, y):

        n_splits=5

        skf = StratifiedShuffleSplit(n_splits=n_splits,test_size=0.2,random_state=0)
        train_accuracy_all=np.zeros((n_splits, self.nb_epochs))
        test_accuracy_all=np.zeros((n_splits , self.nb_epochs))



        compteur=0
        y_test_concat=[]
        y_pred_concat=[]

        for idx_train ,idx_test in skf.split(X, y):
            X_train=X.loc[idx_train]
            Y_train=y.loc[idx_train]
            X_test=X.loc[idx_test]
            Y_test=y.loc[idx_test]

            normalizX=preprocessing.StandardScaler()
            X_train=normalizX.fit_transform(X_train)
            X_test=normalizX.transform(X_test)

    #    normalizY=preprocessing.StandardScaler()
    #    Y_train=normalizY.fit_transform(Y_train)
    #    Y_test=normalizY.transform(Y_test)

            """ Amélioration possible et même a faire: mettre ce qui suit (test_accuracy_all)jusqu'a dans une methode d'une autre classe
            """
            model = Sequential()
            model.add(Dense(units=self.neurons, activation='relu', input_dim=X.shape[1]))
            model.add(Dropout(0.5))
            if self.nb_hidden_layers >0:
                model.add(Dense(units=self.neurons, activation='relu', input_dim=self.neurons))
                model.add(Dropout(0.5))
            if self.nb_hidden_layers >1:
                model.add(Dense(units=self.neurons, activation='relu', input_dim=self.neurons))
                model.add(Dropout(0.5))
            if self.nb_hidden_layers >2:
                model.add(Dense(units=self.gneurons, activation='relu', input_dim=self.neurons))
            model.add(Dense(units=1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy',optimizer=self.optimizer, metrics=['accuracy'])
            for i in range(self.nb_epochs):
                print(i+1,"/",self.nb_epochs)
                res=model.fit(X_train, Y_train, epochs=1, batch_size=self.batch_size)
                score_train=res.history['acc'][0]
                score_test=model.evaluate(X_test,Y_test)
                train_accuracy_all[compteur,i]=score_train
                test_accuracy_all[compteur,i]=score_test[1]
            compteur+=1
            TrainPredictCodeLibelle.saveModel(model)
            y_test_class=pd.DataFrame(Y_test)['statut'].apply(self.numToClass)
            y_test_concat.append(y_test_class)
            y_pred_concat.append(model.predict(X_test, batch_size=128))

        accuracy_model = self.matriceConfusion(y_test_concat, y_pred_concat,train_accuracy_all,test_accuracy_all)

        return accuracy_model
    """
    ToDo: je dois sauvegardé le model avec pickel ou un autre, puis je dois renvoyer
    y_pred_concat
    y_test_concat, et le model sauvegardé
    puis il faudra que la classe prenne en entrée une colonne ou une variable représentant les classes à prédire

"""

#    def get_accuracy_model(self,x,accuracy_model= None):
#        return accuracy_model[x]


    def displayCodelibelle(self, y_test_pred_class, noms_all,y_pred_all, accuracy_model1=None):
        coef_abbatement=0.95
        res_all=pd.DataFrame(noms_all)

        def get_accuracy_model(x,accuracy_model ):
            return accuracy_model[x]

        res_all_accuracy_test=y_test_pred_class.apply(get_accuracy_model, accuracy_model =accuracy_model1)




        res_all['confiance']=y_pred_all[:,0]*100*coef_abbatement*res_all_accuracy_test
        res_all['model_out']=np.round(y_pred_all[:,-1],2)
        res_all['perfo']=np.round(res_all_accuracy_test,2)
        #res_all2=res_all.loc[y_test_pred_class=='lié']
        #if y_test_pred_class in res_all
        #import pdb
        #pdb.set_trace()
        res_all['statut'] = y_test_pred_class

        #import pdb
        #pdb.set_trace()

        #res_all=res_all.loc[y_test_pred_class]
        res_all.to_excel("predColdeLibelle2.xlsx")
        print(res_all)
        #abbat=np.ones(res_all2.shape,dtype='int32')*coef_abbatement

        #res_all2 = spark.createDataFrame(res_all2)

        #res_all2 = res_all2.withColumn('abattement', F.lit(coef_abbatement))

        #res_all2 = res_all2.withColumn('ref_chargement', F.lit(ref_charg))




    def predictCodeLibelle(self, mode=True ):
        #ToDo: compléter cette fonction
        # implémenter les test


        Mymodel = open('myModel.pkl','rb')
        model = joblib.load(Mymodel)
        seuil=0.5
        ListeData = TrainPredictCodeLibelle.readData()
        y = ListeData[0]
        X = ListeData[1]
        noms_all = ListeData[3]
        input_data = ListeData[2]
        if (mode == True):
            accuracy_model_P = self.trainCodeLibelle(y, X)
           # displayCodelibelle()

            y_pred_all=model.predict(input_data, batch_size=128)
            y_pred_all2= (y_pred_all > seuil)*1
            y_test_pred_class=pd.DataFrame(y_pred_all2,columns=['statut'])['statut'].apply(self.numToClass)
            #y_test_pred_class.to_excel("codelibelle.xlsx")
            self.displayCodelibelle(y_test_pred_class, noms_all,y_pred_all, accuracy_model_P )
        return y_test_pred_class
        # ce return est temporaire
        #
