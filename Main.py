# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:17:33 2019

@author: jjohns
"""
path="C:\\Users\\jjohns\\Downloads\\champs-scalar-coupling"
train,test=setup_data.setup_data(path)
import setup_data
import models

train1JHC=train[train['type']=='1JHC']
train1JHN=train[train['type']=='1JHN']
train2JHC=train[train['type']=='2JHC']
train2JHH=train[train['type']=='2JHH']
train2JHN=train[train['type']=='2JHN']
train3JHC=train[train['type']=='3JHC']
train3JHH=train[train['type']=='3JHH']
train3JHN=train[train['type']=='3JHN']
test1JHC=test[test['type']=='1JHC']
test1JHN=test[test['type']=='1JHN']
test2JHC=test[test['type']=='2JHC']
test2JHH=test[test['type']=='2JHH']
test2JHN=test[test['type']=='2JHN']
test3JHC=test[test['type']=='3JHC']
test3JHH=test[test['type']=='3JHH']
test3JHN=test[test['type']=='3JHN']
#%%
model1JHN, J1JHN=models.train_1JHN(train=train1JHN.sample(n=1024, random_state=3), num_epochs=3,batch_size=16, pre_load=False)
