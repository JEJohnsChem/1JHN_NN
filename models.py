# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:10:02 2019

@author: jjohns
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import openbabel as ob
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
#SKlearn Libraries
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
import mol_image_3D as mi3D

#%% Functions Used in models
def rmsle_cv(model, dataset,y):
    """
    Returns the mean absolute error
    """
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dataset)
    rmse= np.log(-cross_val_score(model, dataset, y, scoring="neg_mean_absolute_error", cv = kf))
    return(rmse)
    

def find_path2(mol,atom0_index,atom1_index):
    """
    Find the path between 2 atoms separated by 1 atom
    """
    atom0_index = atom0_index+1
    atom1_index = atom1_index+1
    atom_iter=ob.OBAtomAtomIter(mol.GetAtom(atom0_index))
    alist=[]
    
    index=0
    for a in atom_iter:
        alist.append(a.GetIdx())
        index=index+1
    #print('The list of bound atoms is:', alist)
    index=0
    depth=0
    finished=False
    for atom_index in alist:
        path=atom_index
        atom_iter=ob.OBAtomAtomIter(mol.GetAtom(atom_index))
        for a in atom_iter:
            #print(a.GetIdx())
            if a.GetIdx() ==atom1_index:
                finished=True
                break
            
        if finished:
            break
    if not finished:
        #print('Unable to find a path between atoms',atom0_index-1,' and ',atom1_index-1,'with a depth of 2')
        return -1
    path=path-1
    return path

def find_path3(mol,atom0_index,atom1_index):
   atom0_index = atom0_index+1
   atom1_index = atom1_index+1
   atom_iter=ob.OBAtomAtomIter(mol.GetAtom(atom0_index))
   alist=[]
    
   path=[0 ,0]
   index=0
   for a in atom_iter:
       alist.append(a.GetIdx())
   #print('The list of atoms bound to[',atom0_index,']is:', alist)
   index=0
   depth=0
   finished=False
   for atom_index in alist:
       path[0]=atom_index
       atom_iter=ob.OBAtomAtomIter(mol.GetAtom(atom_index))
       alist2=[]
       for a in atom_iter:
           alist2.append(a.GetIdx())
       #print('The atoms connected to atom',path[0],'are:', alist2)    
       for atom_index2 in alist2:
           path[1]=atom_index2
           atom_iter2=ob.OBAtomAtomIter(mol.GetAtom(atom_index2))
           #print('The atoms connected to',path[1],'are:')
           for a2 in atom_iter2:
               #print(a2.GetIdx())
               if a2.GetIdx() ==atom1_index:
                   finished=True
                   break
           if finished: 
               break
       if finished:
           break
   if not finished:
       print('Unable to find a path between atoms',atom0_index-1,' and ',atom1_index-1,'with a depth of 3')
       return [-1,-1]
   path[0]=path[0]-1
   path[1]=path[1]-1
   return path

#%% Learning 1JHN
def Learn1JHN_CNN(train1JHN, model,fname, file_io='read', path="X:\\CHAMPS\\"):
    if file_io=='write':
        OBConversion=ob.OBConversion()
        OBConversion.SetInFormat("xyz")
        print(len(train1JHN))
        for index in range(0,len(train1JHN)):
            mol=ob.OBMol()
            mol_name=train1JHN.iloc[index]['molecule_name'] +'.xyz'
            OBConversion.ReadFile(mol,mol_name)
            if mol.GetAtom(train1JHN.iloc[index]['atom_index_0'].item()+1).IsNitrogen():
                A=train1JHN.iloc[index]['atom_index_0'].item()+1
                B=train1JHN.iloc[index]['atom_index_1'].item()+1
            else:
                A=train1JHN.iloc[index]['atom_index_1'].item()+1
                B=train1JHN.iloc[index]['atom_index_0'].item()+1
            if index==0:
                X=mi3D.make_conv_input(mol,A,B)
                X=X.reshape((1,64,64,64,2))
                print(X.shape)
            else:
                tmp=mi3D.make_conv_input(mol,A,B)
                X=np.append(X,tmp.reshape(1,64,64,64,2),axis=0)
                print(X.shape)
            if index % 32 ==0:
                print('Molecules 1 - {} made into images'.format(index))
        print('index = {}, fname = {}, file = {}, path = {}'.format(index, fname, file_io, path))
        np.save(path+fname,X)
    else:
        X=np.load(path+fname+".npy")
        print('X loaded successfully')
#    X=(X-np.mean(X))/(np.std(X))
    Y=np.array(train1JHN['scalar_coupling_constant'].reset_index(drop=True))
    Y=Y.reshape((1,len(train1JHN)))
#    print(X.shape)
#    model=keras.Sequential([
#            keras.layers.Dense(256,activation=tf.nn.tanh,input_shape=(64*64*64+6,), kernel_initializer=keras.initializers.he_normal()),
#            keras.layers.Dense(128,activation=tf.nn.tanh,kernel_initializer=keras.initializers.he_normal()),
#            keras.layers.Dense(16,activation=tf.nn.tanh,kernel_initializer=keras.initializers.he_normal()),
#            keras.layers.Dense(1,activation=tf.nn.relu,kernel_initializer=keras.initializers.he_normal())])
    
    
    history=model.fit(X,Y.T,epochs=1, batch_size=32, verbose=2)
#    plt.plot(history.history['mean_absolute_error'])
#    plt.show()
    
    return history, model,X

def train_1JHN(train, num_epochs=10,batch_size=128,width=64,height=64,depth=64,nchannel=2, pre_load=False):
       
    
    num_minibatches=len(train)//batch_size
    J=[]
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3), activation=tf.nn.relu, input_shape=(height,width,depth,2)))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(2,2,2)))
    model.add(tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(2,2,2)))
    model.add(tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(2,2,2)))
    model.add(tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(2,2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
#    model.add(tf.keras.layers.Dense(32,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1,activation=tf.keras.activations.linear))
    model.compile(optimizer='adam', loss='mean_absolute_error',
                      metrics=['mean_absolute_error'])
    model.summary()
    os.chdir('structures')
    print('Summary over')
    for epoch_number in range(num_epochs):
        start=time.time()
        for batch_number in range(num_minibatches):
            if (epoch_number ==0) and (not pre_load):
                history,model,X=Learn1JHN_CNN(train.iloc[batch_number*batch_size : (batch_number+1)*batch_size],model,fname='1JHN'+str(batch_number),file_io='write')
            else:
                history,model,X=Learn1JHN_CNN(train.iloc[batch_number*batch_size : (batch_number+1)*batch_size],model,fname='1JHN'+str(batch_number),file_io='read')
            print('Minibatch # {}'.format(batch_number+1))
            J.append(history.history['mean_absolute_error'])
            if batch_number % 2 ==0:
                plt.plot(J)
                plt.show()
                print('Functional Time to do 5 minibatches is {}'.format(time.time()-start))
        print('Finishing epoch number {}'.format(epoch_number))
        os.chdir('..\\')
    return model, J