# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:36:04 2019

@author: jjohns
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 08:33:54 2019

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
lasso=make_pipeline(RobustScaler(),Lasso(alpha=0.05))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

n_folds = 5

def rmsle_cv(model, dataset,y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dataset)
    rmse= np.log(-cross_val_score(model, dataset, y, scoring="neg_mean_absolute_error", cv = kf))
    return(rmse)


#Define some path finding functions


def find_path2(mol,atom0_index,atom1_index):
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
#%%## 2JHN Couplings
#A is the Nitrogen index+1, B is the connecting atom index+1, C=Hydrogen atom index+1
def Learn2JHN(train2JHN, test2JHN):
    
    n2JHN=len(test2JHN)+len(train2JHN)
    start=time.time()

    AB_Distance=np.zeros(len(train2JHN))
    BC_Distance=np.zeros(len(train2JHN))
    AC_Distance=np.zeros(len(train2JHN))
    ABC_Distance=np.zeros(len(train2JHN))
    ABC_Angle=np.zeros(len(train2JHN))
    A_Aromatic=[]
    A_Hybrid=np.zeros(len(train2JHN))
    A_Valence=np.zeros(len(train2JHN))
    ABC_Angle=np.zeros(len(train2JHN))
    B_Atomic_Num=np.zeros(len(train2JHN))
    B_Hybrid=np.zeros(len(train2JHN))
    B_Valence=np.zeros(len(train2JHN))
    B_Aromatic=[]
    lasso=make_pipeline(RobustScaler(),Lasso(alpha=0.05))
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
    
    n_folds = 5

    def rmsle_cv(model, dataset,y):
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dataset)
        rmse= np.log(-cross_val_score(model, dataset, y, scoring="neg_mean_absolute_error", cv = kf))
        return(rmse)
    
    OBConversion=ob.OBConversion()
    OBConversion.SetInFormat("xyz")
    for index in range(0,len(train2JHN)):
        mol=ob.OBMol()
        mol_name=train2JHN.iloc[index]['molecule_name'] +'.xyz'
        OBConversion.ReadFile(mol,mol_name)
        if mol.GetAtom(train2JHN.iloc[index]['atom_index_0'].item()+1).IsNitrogen():
            A=train2JHN.iloc[index]['atom_index_0'].item()+1
            C=train2JHN.iloc[index]['atom_index_1'].item()+1
        else:
            A=train2JHN.iloc[index]['atom_index_1'].item()+1
            C=train2JHN.iloc[index]['atom_index_0'].item()+1
        B=find_path2(mol,train2JHN.iloc[index]['atom_index_0'].item(),train2JHN.iloc[index]['atom_index_1'].item())+1
        AB_Distance[index]=mol.GetAtom(A).GetDistance(B)
        AC_Distance[index]=mol.GetAtom(A).GetDistance(C)
        BC_Distance[index]=mol.GetAtom(B).GetDistance(C)
        ABC_Distance[index]=AB_Distance[index]+BC_Distance[index]
        ABC_Angle[index]=np.deg2rad(mol.GetAtom(A).GetAngle(B,C))
        A_Hybrid[index]=mol.GetAtom(A).GetHyb()
        A_Valence[index]=mol.GetAtom(A).GetValence()
        A_Aromatic.append(mol.GetAtom(A).IsAromatic())
        B_Atomic_Num[index]=mol.GetAtom(B).GetAtomicNum()
        B_Hybrid[index]=mol.GetAtom(B).GetHyb()
        B_Valence[index]=mol.GetAtom(B).GetValence()
        B_Aromatic.append(mol.GetAtom(B).IsAromatic())
        

    df2JHN=pd.DataFrame({'AB_Distance':AB_Distance, 'BC_Distance': BC_Distance, 'AC_Distance': AC_Distance,  'ABC_Distance': ABC_Distance, \
                         'ABC_Angle':ABC_Angle, 'A_Hybrid':A_Hybrid, 'A_Valence':A_Valence, 'A_Aromatic':A_Aromatic, 'B_Atomic_Num': B_Atomic_Num, \
                         'B_Hybrid': B_Hybrid, 'B_Valence': B_Valence, 'B_Aromatic':B_Aromatic})    
                       
    y2JHN=train2JHN['scalar_coupling_constant'].reset_index(drop=True)
    
    plt.hist(AC_Distance)
    plt.xlabel('Distance between H and N.  Should be roughly 2 A')
    plt.ylabel('Counts')
    plt.ion()
    
    plt.hist(ABC_Distance)
    plt.xlabel('Topological Distance between H and N.  Should be roughly 3 A')
    plt.ylabel('Counts')
    plt.ion()
    
    plt.hist(ABC_Angle)
    plt.ylabel('Counts')
    plt.xlabel('Angle from A to B to C')
    plt.ion()
    
    fig, ax = plt.subplots()
    ax.scatter(x=ABC_Distance,y=train2JHN['scalar_coupling_constant'])
    ax.set(xlabel='Bond Distance in Angsroms', ylabel='NH Scalar Coupling Constant',title='2JHN vs Topological Distance')
    plt.ion()
    
    fig, ax = plt.subplots()
    ax.scatter(x=ABC_Angle, y=train2JHN['scalar_coupling_constant'])
    ax.set(xlabel='Molecular Angle in Radians', ylabel='NH Scalar Coupling Constant',title='2JHN vs Angle')
    plt.ion()
    
    
    AB_Distance=np.zeros(len(test2JHN))
    BC_Distance=np.zeros(len(test2JHN))
    AC_Distance=np.zeros(len(test2JHN))
    ABC_Distance=np.zeros(len(test2JHN))
    ABC_Angle=np.zeros(len(test2JHN))
    A_Aromatic=[]
    A_Hybrid=np.zeros(len(test2JHN))
    A_Valence=np.zeros(len(test2JHN))
    ABC_Angle=np.zeros(len(test2JHN))
    B_Atomic_Num=np.zeros(len(test2JHN))
    B_Hybrid=np.zeros(len(test2JHN))
    B_Valence=np.zeros(len(test2JHN))
    B_Aromatic=[]
    
    for index in range(0,len(test2JHN)):
        mol=ob.OBMol()
        mol_name=test2JHN.iloc[index]['molecule_name'] +'.xyz'
        OBConversion.ReadFile(mol, mol_name)
        if mol.GetAtom(test2JHN.iloc[index]['atom_index_0'].item()+1).IsNitrogen():
            A=test2JHN.iloc[index]['atom_index_0'].item()+1
            C=test2JHN.iloc[index]['atom_index_1'].item()+1
        else:
            A=test2JHN.iloc[index]['atom_index_1'].item()+1
            C=test2JHN.iloc[index]['atom_index_0'].item()+1
        B=find_path2(mol,test2JHN.iloc[index]['atom_index_0'].item(),test2JHN.iloc[index]['atom_index_1'].item())+1
        AB_Distance[index]=mol.GetAtom(A).GetDistance(B)
        AC_Distance[index]=mol.GetAtom(A).GetDistance(C)
        BC_Distance[index]=mol.GetAtom(B).GetDistance(C)
        ABC_Distance[index]=AB_Distance[index] + BC_Distance[index]
        ABC_Angle[index]=np.deg2rad(mol.GetAtom(A).GetAngle(B,C))
        A_Hybrid[index]=mol.GetAtom(A).GetHyb()
        A_Valence[index]=mol.GetAtom(A).GetValence()
        A_Aromatic.append(mol.GetAtom(A).IsAromatic())
        B_Atomic_Num[index]=mol.GetAtom(B).GetAtomicNum()
        B_Hybrid[index]=mol.GetAtom(B).GetHyb()
        B_Valence[index]=mol.GetAtom(B).GetValence()
        B_Aromatic.append(mol.GetAtom(B).IsAromatic())
    
    xtest2JHN=pd.DataFrame({'AB_Distance':AB_Distance, 'BC_Distance': BC_Distance, 'AC_Distance': AC_Distance,  'ABC_Distance': ABC_Distance, \
                         'ABC_Angle':ABC_Angle, 'A_Hybrid':A_Hybrid, 'A_Valence':A_Valence, 'A_Aromatic':A_Aromatic, 'B_Atomic_Num': B_Atomic_Num, \
                         'B_Hybrid': B_Hybrid, 'B_Valence': B_Valence, 'B_Aromatic':B_Aromatic}) 

    stop=time.time()
    print('It took', stop-start,'seconds to run through',n2JHN,'2JHN couplings in the dataset')
    start=time.time()
    lasso.fit(pd.get_dummies(df2JHN),y2JHN)
    stop=time.time()
    print('It took', stop-start, 'seconds to train the model on', len(train2JHN),'examples')
    pred2JHN=lasso.predict(pd.get_dummies(xtest2JHN))



    y_pred=lasso.predict(df2JHN)
    fig, ax = plt.subplots()
    ax.scatter(x=y2JHN, y=y_pred)
    ax.set(xlabel='Actual 2JHN Values', ylabel='Predicted 2JHN Coupling Constants',title='Model vs Real')
    plt.ion()
    print(pred2JHN[0:3])
    
    score = rmsle_cv(model_xgb,df2JHN.values,y2JHN)
    print("Xgboost 2JHN score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    
    model_xgb.fit(df2JHN.values, y2JHN)
    pred2=model_xgb.predict(df2JHN.values)
    
    fig, ax = plt.subplots()
    ax.scatter(x=y2JHN, y=pred2)
    ax.set(xlabel='Actual 2JHN Values', ylabel='XGBoost Predicted 2JHN Coupling Constants',title='XGBoost Model vs Real')
    plt.ion()
    submission2JHN=pd.DataFrame({'id': test2JHN['id'], 'scalar_coupling_constant': model_xgb.predict(xtest2JHN.values)})
    submission2JHN.to_csv('submission2JHN.csv', index=False)
    score = rmsle_cv(model_xgb,df2JHN.values,y2JHN)
    print("Xgboost 2JHN score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#%% 
## 3JHN Couplings
#A is the C index+1, B is the connecting atom index+1, C=Connecting atom+1, D=H+1
def Learn3JHN(train3JHN, test3JHN):
            
    n3JHN=len(test3JHN)+len(train3JHN)
    start=time.time()
    
    AB_Distance=np.zeros(len(train3JHN))
    BC_Distance=np.zeros(len(train3JHN))
    CD_Distance=np.zeros(len(train3JHN))
    AC_Distance=np.zeros(len(train3JHN))
    AD_Distance=np.zeros(len(train3JHN))
    BD_Distance=np.zeros(len(train3JHN))
    ABC_Distance=np.zeros(len(train3JHN))
    BCD_Distance=np.zeros(len(train3JHN))
    ABCD_Distance=np.zeros(len(train3JHN))
    ABC_Angle=np.zeros(len(train3JHN))
    BCD_Angle=np.zeros(len(train3JHN))
    ABCD_Dihedral=np.zeros(len(train3JHN))
    A_Aromatic=[]
    A_Hybrid=np.zeros(len(train3JHN))
    A_Valence=np.zeros(len(train3JHN))
    ABC_Angle=np.zeros(len(train3JHN))
    B_Atomic_Num=np.zeros(len(train3JHN))
    B_Hybrid=np.zeros(len(train3JHN))
    B_Valence=np.zeros(len(train3JHN))
    B_Aromatic=[]
    C_Atomic_Num=np.zeros(len(train3JHN))
    C_Hybrid=np.zeros(len(train3JHN))
    C_Valence=np.zeros(len(train3JHN))
    C_Aromatic=[]
    lasso=make_pipeline(RobustScaler(),Lasso(alpha=0.05))
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
    
    n_folds = 5

    def rmsle_cv(model, dataset,y):
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dataset)
        rmse= np.log(-cross_val_score(model, dataset, y, scoring="neg_mean_absolute_error", cv = kf))
        return(rmse)
    OBConversion=ob.OBConversion()
    OBConversion.SetInFormat("xyz")
    for index in range(0,len(train3JHN)):
        mol_name=train3JHN.iloc[index]['molecule_name'] +'.xyz'
        mol=ob.OBMol()
        OBConversion.ReadFile(mol, mol_name)
        if mol.GetAtom(train3JHN.iloc[index]['atom_index_0'].item()+1).IsNitrogen():
            A=train3JHN.iloc[index]['atom_index_0'].item()+1
            D=train3JHN.iloc[index]['atom_index_1'].item()+1
        else:
            A=train3JHN.iloc[index]['atom_index_1'].item()+1
            D=train3JHN.iloc[index]['atom_index_0'].item()+1
        path=find_path3(mol,A-1,D-1)
        B=path[0]+1
        C=path[1]+1
        AB_Distance[index]=mol.GetAtom(A).GetDistance(B)
        AC_Distance[index]=mol.GetAtom(A).GetDistance(C)
        BC_Distance[index]=mol.GetAtom(B).GetDistance(C)
        CD_Distance[index]=mol.GetAtom(C).GetDistance(D)
        AD_Distance[index]=mol.GetAtom(A).GetDistance(D)
        BD_Distance[index]=mol.GetAtom(B).GetDistance(D)
        
        ABC_Distance[index]=AB_Distance[index]+BC_Distance[index]
        BCD_Distance[index]=BC_Distance[index]+CD_Distance[index]
        ABCD_Distance[index]=ABC_Distance[index]+CD_Distance[index]
        ABC_Angle[index]=np.deg2rad(mol.GetAtom(A).GetAngle(B,C))
        BCD_Angle[index]=np.deg2rad(mol.GetAtom(B).GetAngle(C,D))
        ABCD_Dihedral[index]=np.deg2rad(mol.GetTorsion(A,B,C,D))
        A_Hybrid[index]=mol.GetAtom(A).GetHyb()
        A_Valence[index]=mol.GetAtom(A).GetValence()
        A_Aromatic.append(mol.GetAtom(A).IsAromatic())
        B_Atomic_Num[index]=mol.GetAtom(B).GetAtomicNum()
        B_Hybrid[index]=mol.GetAtom(B).GetHyb()
        B_Valence[index]=mol.GetAtom(B).GetValence()
        B_Aromatic.append(mol.GetAtom(B).IsAromatic())
        C_Atomic_Num[index]=mol.GetAtom(C).GetAtomicNum()
        C_Hybrid[index]=mol.GetAtom(C).GetHyb()
        C_Valence[index]=mol.GetAtom(C).GetValence()
        C_Aromatic.append(mol.GetAtom(C).IsAromatic())

        

    df3JHN=pd.DataFrame({'AB_Distance':AB_Distance, 'AC_Distance': AC_Distance, 'AD_Distance': AD_Distance, 'BC_Distance':BC_Distance,\
                         'BD_Distance':BD_Distance, 'CD_Distance': CD_Distance, 'ABC_Distance':ABC_Distance, 'BCD_Distance':BCD_Distance,\
                         'ABCD_Distance': ABCD_Distance,'ABC_Angle':ABC_Angle, 'BCD_Angle':BCD_Angle, 'ABCD_Dihedral':ABCD_Dihedral, \
                         'A_Hybrid':A_Hybrid, 'A_Valence':A_Valence, 'A_Aromatic':A_Aromatic, 'B_Atomic_Num': B_Atomic_Num, \
                         'B_Hybrid': B_Hybrid, 'B_Valence': B_Valence, 'B_Aromatic':B_Aromatic,'C_Atomic_Num':C_Atomic_Num,'C_Hybrid':C_Hybrid,\
                         'C_Valence':C_Valence, 'C_Aromatic':C_Aromatic, 'Cos_Theta':np.cos(ABCD_Dihedral),\
                         'Cos_2Theta':np.cos(2*ABCD_Dihedral)})                       
    y3JHN=train3JHN['scalar_coupling_constant'].reset_index(drop=True)
    
    plt.hist(AD_Distance)
    plt.xlabel('Distance between H and N.  Should be roughly 2 A')
    plt.ylabel('Counts')
    plt.ion()
    
    plt.hist(ABCD_Distance)
    plt.xlabel('Topological Distance between H and N.  Should be roughly 3 A')
    plt.ylabel('Counts')
    plt.ion()
    
    plt.hist(ABCD_Dihedral)
    plt.ylabel('Counts')
    plt.xlabel('Dihedral Angle from A to B to C to D')
    plt.ion()
    
    fig, ax = plt.subplots()
    ax.scatter(x=ABCD_Distance,y=train3JHN['scalar_coupling_constant'])
    ax.set(xlabel='Bond Distance in Angsroms', ylabel='NH Scalar Coupling Constant',title='3JHN vs Topological Distance')
    plt.ion()
    
    fig, ax = plt.subplots()
    ax.scatter(x=ABCD_Dihedral, y=train3JHN['scalar_coupling_constant'])
    ax.set(xlabel='Molecular Angle in Radians', ylabel='NH Scalar Coupling Constant',title='3JHN vs Angle')
    plt.ion()


    AB_Distance=np.zeros(len(test3JHN))
    BC_Distance=np.zeros(len(test3JHN))
    CD_Distance=np.zeros(len(test3JHN))
    AC_Distance=np.zeros(len(test3JHN))
    AD_Distance=np.zeros(len(test3JHN))
    BD_Distance=np.zeros(len(test3JHN))
    ABC_Distance=np.zeros(len(test3JHN))
    BCD_Distance=np.zeros(len(test3JHN))
    ABCD_Distance=np.zeros(len(test3JHN))
    ABC_Angle=np.zeros(len(test3JHN))
    BCD_Angle=np.zeros(len(test3JHN))
    ABCD_Dihedral=np.zeros(len(test3JHN))
    A_Aromatic=[]
    A_Hybrid=np.zeros(len(test3JHN))
    A_Valence=np.zeros(len(test3JHN))
    ABC_Angle=np.zeros(len(test3JHN))
    B_Atomic_Num=np.zeros(len(test3JHN))
    B_Hybrid=np.zeros(len(test3JHN))
    B_Valence=np.zeros(len(test3JHN))
    B_Aromatic=[]
    C_Atomic_Num=np.zeros(len(test3JHN))
    C_Hybrid=np.zeros(len(test3JHN))
    C_Valence=np.zeros(len(test3JHN))
    C_Aromatic=[]
    
    
    for index in range(0,len(test3JHN)):
        mol_name=test3JHN.iloc[index]['molecule_name'] +'.xyz'
        mol=ob.OBMol()
        OBConversion.ReadFile(mol,mol_name)
        if mol.GetAtom(test3JHN.iloc[index]['atom_index_0'].item()+1).IsCarbon():
            A=test3JHN.iloc[index]['atom_index_0'].item()+1
            D=test3JHN.iloc[index]['atom_index_1'].item()+1
        else:
            A=test3JHN.iloc[index]['atom_index_1'].item()+1
            D=test3JHN.iloc[index]['atom_index_0'].item()+1
        path=find_path3(mol,A-1,D-1)
        B=path[0]+1
        C=path[1]+1
        AB_Distance[index]=mol.GetAtom(A).GetDistance(B)
        AC_Distance[index]=mol.GetAtom(A).GetDistance(C)
        BC_Distance[index]=mol.GetAtom(B).GetDistance(C)
        CD_Distance[index]=mol.GetAtom(C).GetDistance(D)
        AD_Distance[index]=mol.GetAtom(A).GetDistance(D)
        BD_Distance[index]=mol.GetAtom(B).GetDistance(D)
        
        ABC_Distance[index]=AB_Distance[index]+BC_Distance[index]
        BCD_Distance[index]=BC_Distance[index]+CD_Distance[index]
        ABCD_Distance[index]=ABC_Distance[index]+CD_Distance[index]
        ABC_Angle[index]=np.deg2rad(mol.GetAtom(A).GetAngle(B,C))
        BCD_Angle[index]=np.deg2rad(mol.GetAtom(B).GetAngle(C,D))
        ABCD_Dihedral[index]=np.deg2rad(mol.GetTorsion(A,B,C,D))
        A_Hybrid[index]=mol.GetAtom(A).GetHyb()
        A_Valence[index]=mol.GetAtom(A).GetValence()
        A_Aromatic.append(mol.GetAtom(A).IsAromatic())
        B_Atomic_Num[index]=mol.GetAtom(B).GetAtomicNum()
        B_Hybrid[index]=mol.GetAtom(B).GetHyb()
        B_Valence[index]=mol.GetAtom(B).GetValence()
        B_Aromatic.append(mol.GetAtom(B).IsAromatic())
        C_Atomic_Num[index]=mol.GetAtom(C).GetAtomicNum()
        C_Hybrid[index]=mol.GetAtom(C).GetHyb()
        C_Valence[index]=mol.GetAtom(C).GetValence()
        C_Aromatic.append(mol.GetAtom(C).IsAromatic())
    
            
    
    xtest3JHN=pd.DataFrame({'AB_Distance':AB_Distance, 'AC_Distance': AC_Distance, 'AD_Distance': AD_Distance, 'BC_Distance':BC_Distance,\
                         'BD_Distance':BD_Distance, 'CD_Distance': CD_Distance, 'ABC_Distance':ABC_Distance, 'BCD_Distance':BCD_Distance,\
                         'ABCD_Distance': ABCD_Distance,'ABC_Angle':ABC_Angle, 'BCD_Angle':BCD_Angle, 'ABCD_Dihedral':ABCD_Dihedral, \
                         'A_Hybrid':A_Hybrid, 'A_Valence':A_Valence, 'A_Aromatic':A_Aromatic, 'B_Atomic_Num': B_Atomic_Num, \
                         'B_Hybrid': B_Hybrid, 'B_Valence': B_Valence, 'B_Aromatic':B_Aromatic,'C_Atomic_Num':C_Atomic_Num,'C_Hybrid':C_Hybrid,\
                         'C_Valence':C_Valence, 'C_Aromatic':C_Aromatic, 'Cos_Theta':np.cos(ABCD_Dihedral),\
                         'Cos_2Theta':np.cos(2*ABCD_Dihedral)}) 
    
    
    stop=time.time()
    print('It took', stop-start,'seconds to run through',n3JHN,'3JHN couplings in the dataset')
    start=time.time()
    lasso.fit(pd.get_dummies(df3JHN),y3JHN)
    stop=time.time()
    print('It took', stop-start, 'seconds to train the model on', len(train3JHN),'examples')
    pred3JHN=lasso.predict(pd.get_dummies(xtest3JHN))
    
    
    
    y_pred=lasso.predict(df3JHN)
    fig, ax = plt.subplots()
    ax.scatter(x=y3JHN, y=y_pred)
    ax.set(xlabel='Actual 3JHN Values', ylabel='Predicted 3JHN Coupling Constants',title='Model vs Real')
    plt.ion()
    print(pred3JHN[0:3])
    

    
    start=time.time()
    model_xgb.fit(df3JHN.values, y3JHN)
    pred2=model_xgb.predict(df3JHN.values)
    stop=time.time()
    print('It took', stop-start,'seconds to train xgboost for 3JHN Couplings')
    fig, ax = plt.subplots()
    ax.scatter(x=y3JHN, y=pred2)
    ax.set(xlabel='Actual 3JHN Values', ylabel='XGBoost Predicted 3JHN Coupling Constants',title='XGBoost Model vs Real')
    plt.ion()
    submission3JHN=pd.DataFrame({'id': test3JHN['id'], 'scalar_coupling_constant': model_xgb.predict(xtest3JHN.values)})
    submission3JHN.to_csv('submission3JHN.csv', index=False)
    score = rmsle_cv(model_xgb,df3JHN.values,y3JHN)
    print("Xgboost 3JHN score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#%%1JHN
def Learn1JHN(train1JHN, test1JHN):
    
    n1JHN=len(test1JHN)+len(train1JHN)
    start=time.time()

    AB_Distance=np.zeros(len(train1JHN))
    A_Aromatic=[]
    A_Hybrid=np.zeros(len(train1JHN))
    A_Valence=np.zeros(len(train1JHN))
    A_NearestNeighbors=np.zeros(len(train1JHN))
    A1_Atomic_Num=np.zeros(len(train1JHN))
    A1_Hybrid=np.zeros(len(train1JHN))
    A1_Valence=np.zeros(len(train1JHN))
    A1_Aromatic=[]
    A1_Distance=np.zeros(len(train1JHN))
    BAA1_Angle=np.zeros(len(train1JHN))
    A2_Atomic_Num=np.zeros(len(train1JHN))
    A2_Hybrid=np.zeros(len(train1JHN))
    A2_Valence=np.zeros(len(train1JHN))
    A2_Aromatic=[]
    A2_Distance=np.zeros(len(train1JHN))
    BAA2_Angle=np.zeros(len(train1JHN))
    A3_Atomic_Num=np.zeros(len(train1JHN))
    A3_Hybrid=np.zeros(len(train1JHN))
    A3_Valence=np.zeros(len(train1JHN))
    A3_Aromatic=[]
    A3_Distance=np.zeros(len(train1JHN))
    BAA3_Angle=np.zeros(len(train1JHN))
    A1AA2_Angle=np.zeros(len(train1JHN))
    A1AA3_Angle=np.zeros(len(train1JHN))
    A2AA3_Angle=np.zeros(len(train1JHN))
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
    
    n_folds = 5

    def rmsle_cv(model, dataset,y):
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dataset)
        rmse= np.log(-cross_val_score(model, dataset, y, scoring="neg_mean_absolute_error", cv = kf))
        return(rmse)
    
    OBConversion=ob.OBConversion()
    OBConversion.SetInFormat("xyz")
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
        
        AB_Distance[index]=mol.GetAtom(A).GetDistance(B)
        A_Hybrid[index]=mol.GetAtom(A).GetHyb()
        A_Valence[index]=mol.GetAtom(A).GetValence()
        A_Aromatic.append(mol.GetAtom(A).IsAromatic())
        atom_iter=ob.OBAtomAtomIter(mol.GetAtom(A))
        alist=[]
        for a in atom_iter:
            alist.append(a.GetIdx())
        A_NearestNeighbors[index]=len(alist)    
        alist.remove(B)
        if alist ==[]:
            A1_Aromatic.append(None)
            A2_Aromatic.append(None)
            A3_Aromatic.append(None)
        elif len(alist)==1:
            A1_Atomic_Num[index]=mol.GetAtom(alist[0]).GetAtomicNum()
            A1_Hybrid[index]=mol.GetAtom(alist[0]).GetHyb()
            A1_Valence[index]=mol.GetAtom(alist[0]).GetValence()
            A1_Aromatic.append(mol.GetAtom(alist[0]).IsAromatic())
            A1_Distance[index]=mol.GetAtom(alist[0]).GetDistance(A)
            BAA1_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[0])
            A2_Atomic_Num[index]=None
            A2_Hybrid[index]=None
            A2_Valence[index]=None
            A2_Aromatic.append(None)
            A2_Distance[index]=None
            BAA2_Angle[index]=None
            A3_Atomic_Num[index]=None
            A3_Hybrid[index]=None
            A3_Valence[index]=None
            A3_Aromatic.append(None)
            A3_Distance[index]=None
            BAA3_Angle[index]=None
            A1AA2_Angle[index]=None
            A1AA3_Angle[index]=None
            A2AA3_Angle[index]=None
        elif len(alist)==2:
            A1_Atomic_Num[index]=mol.GetAtom(alist[0]).GetAtomicNum()
            A1_Hybrid[index]=mol.GetAtom(alist[0]).GetHyb()
            A1_Valence[index]=mol.GetAtom(alist[0]).GetValence()
            A1_Aromatic.append(mol.GetAtom(alist[0]).IsAromatic())
            A1_Distance[index]=mol.GetAtom(alist[0]).GetDistance(A)
            BAA1_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[0])
            A2_Atomic_Num[index]=mol.GetAtom(alist[1]).GetAtomicNum()
            A2_Hybrid[index]=mol.GetAtom(alist[1]).GetHyb()
            A2_Valence[index]=mol.GetAtom(alist[1]).GetValence()
            A2_Aromatic.append(mol.GetAtom(alist[1]).IsAromatic())
            A2_Distance[index]=mol.GetAtom(alist[1]).GetDistance(A)
            BAA2_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[1])
            A3_Atomic_Num[index]=None
            A3_Hybrid[index]=None
            A3_Valence[index]=None
            A3_Aromatic.append(None)
            A3_Distance[index]=None            
            BAA3_Angle[index]=None
            A1AA2_Angle[index]=mol.GetAtom(alist[0]).GetAngle(A,alist[1])
            A1AA3_Angle[index]=None
            A2AA3_Angle[index]=None
        elif len(alist)==3:
            A1_Atomic_Num[index]=mol.GetAtom(alist[0]).GetAtomicNum()
            A1_Hybrid[index]=mol.GetAtom(alist[0]).GetHyb()
            A1_Valence[index]=mol.GetAtom(alist[0]).GetValence()
            A1_Aromatic.append(mol.GetAtom(alist[0]).IsAromatic())
            A1_Distance[index]=mol.GetAtom(alist[0]).GetDistance(A)
            BAA1_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[0])
            A2_Atomic_Num[index]=mol.GetAtom(alist[1]).GetAtomicNum()
            A2_Hybrid[index]=mol.GetAtom(alist[1]).GetHyb()
            A2_Valence[index]=mol.GetAtom(alist[1]).GetValence()
            A2_Aromatic.append(mol.GetAtom(alist[1]).IsAromatic())
            A2_Distance[index]=mol.GetAtom(alist[1]).GetDistance(A)
            BAA2_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[1])
            A3_Atomic_Num[index]=mol.GetAtom(alist[1]).GetAtomicNum()
            A3_Hybrid[index]=mol.GetAtom(alist[2]).GetHyb()
            A3_Valence[index]=mol.GetAtom(alist[2]).GetValence()
            A3_Aromatic.append(mol.GetAtom(alist[2]).IsAromatic())
            A3_Distance[index]=mol.GetAtom(alist[2]).GetDistance(A)
            BAA3_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[2])
            A1AA2_Angle[index]=mol.GetAtom(alist[0]).GetAngle(A,alist[1])
            A1AA3_Angle[index]=mol.GetAtom(alist[0]).GetAngle(A,alist[2])
            A2AA3_Angle[index]=mol.GetAtom(alist[1]).GetAngle(A,alist[2])
        else:
            print('The len of alist is weird: ',len(alist), '.  The index of the weirdo is:' ,index)
            print('alist = ',alist)
            print('A=', A, 'B=', B)
        if index==0:
            print(A2_Aromatic)
            
    print(len(A1_Aromatic), len(A2_Aromatic), len(A1_Hybrid), len(A_NearestNeighbors),len(A3_Distance)) 

    df1JHN=pd.DataFrame({'AB_Distance':AB_Distance, 'A_Hybrid':A_Hybrid, 'A_Valence':A_Valence, 'A_Aromatic':A_Aromatic, \
                         'A_NearestNeighbors':A_NearestNeighbors, 'A1_Atomic_Num':A1_Atomic_Num, 'A1_Hybrid':A1_Hybrid,\
                         'A1_Valence':A1_Valence, 'A1_Aromatic':A1_Aromatic, 'A1_Distance':A1_Distance, 'BAA1_Angle':BAA1_Angle, \
                         'A2_Atomic_Num': A2_Atomic_Num,'A2_Hybrid': A2_Hybrid, 'A2_Valence':A2_Valence, 'A2_Aromatic':A2_Aromatic,\
                         'A2_Distance':A2_Distance, 'BAA2_Angle':BAA2_Angle,'A3_Atomic_Num':A3_Atomic_Num, 'A3_Hybrid':A3_Hybrid, \
                         'A3_Valence':A3_Valence, 'A3_Aromatic':A3_Aromatic, 'A3_Distance':A3_Distance, 'BAA3_Angle':BAA3_Angle,
                         'A1AA2_Angle':A1AA2_Angle, 'A1AA3_Angle':A1AA3_Angle, 'A2AA3_Angle':A2AA3_Angle})    
                       
    y1JHN=train1JHN['scalar_coupling_constant'].reset_index(drop=True)
    
    plt.hist(AB_Distance)
    plt.xlabel('Distance between H and N.  Should be roughly 2 A')
    plt.ylabel('Counts')
    plt.ion()
    
   
    
    fig, ax = plt.subplots()
    ax.scatter(x=AB_Distance,y=train1JHN['scalar_coupling_constant'])
    ax.set(xlabel='Bond Distance in Angsroms', ylabel='NH Scalar Coupling Constant',title='1JHN vs Topological Distance')
    plt.ion()
    
   
    
    
    AB_Distance=np.zeros(len(test1JHN))
    A_Aromatic=[]
    A_Hybrid=np.zeros(len(test1JHN))
    A_Valence=np.zeros(len(test1JHN))
    A_NearestNeighbors=np.zeros(len(test1JHN))
    A1_Atomic_Num=np.zeros(len(test1JHN))
    A1_Hybrid=np.zeros(len(test1JHN))
    A1_Valence=np.zeros(len(test1JHN))
    A1_Aromatic=[]
    A1_Distance=np.zeros(len(test1JHN))
    BAA1_Angle=np.zeros(len(test1JHN))
    A2_Atomic_Num=np.zeros(len(test1JHN))
    A2_Hybrid=np.zeros(len(test1JHN))
    A2_Valence=np.zeros(len(test1JHN))
    A2_Aromatic=[]
    A2_Distance=np.zeros(len(test1JHN))
    BAA2_Angle=np.zeros(len(test1JHN))
    A3_Atomic_Num=np.zeros(len(test1JHN))
    A3_Hybrid=np.zeros(len(test1JHN))
    A3_Valence=np.zeros(len(test1JHN))
    A3_Aromatic=[]
    A3_Distance=np.zeros(len(test1JHN))
    BAA3_Angle=np.zeros(len(test1JHN))
    A1AA2_Angle=np.zeros(len(test1JHN))
    A1AA3_Angle=np.zeros(len(test1JHN))
    A2AA3_Angle=np.zeros(len(test1JHN))
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
    
    n_folds = 5

    def rmsle_cv(model, dataset,y):
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dataset)
        rmse= np.log(-cross_val_score(model, dataset, y, scoring="neg_mean_absolute_error", cv = kf))
        return(rmse)
    
    OBConversion=ob.OBConversion()
    OBConversion.SetInFormat("xyz")
    for index in range(0,len(test1JHN)):
        mol=ob.OBMol()
        mol_name=test1JHN.iloc[index]['molecule_name'] +'.xyz'
        OBConversion.ReadFile(mol,mol_name)
        if mol.GetAtom(test1JHN.iloc[index]['atom_index_0'].item()+1).IsNitrogen():
            A=test1JHN.iloc[index]['atom_index_0'].item()+1
            B=test1JHN.iloc[index]['atom_index_1'].item()+1
        else:
            A=test1JHN.iloc[index]['atom_index_1'].item()+1
            B=test1JHN.iloc[index]['atom_index_0'].item()+1
        
        AB_Distance[index]=mol.GetAtom(A).GetDistance(B)
        A_Hybrid[index]=mol.GetAtom(A).GetHyb()
        A_Valence[index]=mol.GetAtom(A).GetValence()
        A_Aromatic.append(mol.GetAtom(A).IsAromatic())
        atom_iter=ob.OBAtomAtomIter(mol.GetAtom(A))
        alist=[]
        for a in atom_iter:
            alist.append(a.GetIdx())
        A_NearestNeighbors[index]=len(alist)    
        alist.remove(B)
        if alist ==[]:
            A1_Aromatic.append(None)
            A2_Aromatic.append(None)
            A3_Aromatic.append(None)
        elif len(alist)==1:
            A1_Atomic_Num[index]=mol.GetAtom(alist[0]).GetAtomicNum()
            A1_Hybrid[index]=mol.GetAtom(alist[0]).GetHyb()
            A1_Valence[index]=mol.GetAtom(alist[0]).GetValence()
            A1_Aromatic.append(mol.GetAtom(alist[0]).IsAromatic())
            A1_Distance[index]=mol.GetAtom(alist[0]).GetDistance(A)
            BAA1_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[0])
            A2_Atomic_Num[index]=None
            A2_Hybrid[index]=None
            A2_Valence[index]=None
            A2_Aromatic.append(None)
            A2_Distance[index]=None
            BAA2_Angle[index]=None
            A3_Atomic_Num[index]=None
            A3_Hybrid[index]=None
            A3_Valence[index]=None
            A3_Aromatic.append(None)
            A3_Distance[index]=None
            BAA3_Angle[index]=None
            A1AA2_Angle[index]=None
            A1AA3_Angle[index]=None
            A2AA3_Angle[index]=None
        elif len(alist)==2:
            A1_Atomic_Num[index]=mol.GetAtom(alist[0]).GetAtomicNum()
            A1_Hybrid[index]=mol.GetAtom(alist[0]).GetHyb()
            A1_Valence[index]=mol.GetAtom(alist[0]).GetValence()
            A1_Aromatic.append(mol.GetAtom(alist[0]).IsAromatic())
            A1_Distance[index]=mol.GetAtom(alist[0]).GetDistance(A)
            BAA1_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[0])
            A2_Atomic_Num[index]=mol.GetAtom(alist[1]).GetAtomicNum()
            A2_Hybrid[index]=mol.GetAtom(alist[1]).GetHyb()
            A2_Valence[index]=mol.GetAtom(alist[1]).GetValence()
            A2_Aromatic.append(mol.GetAtom(alist[1]).IsAromatic())
            A2_Distance[index]=mol.GetAtom(alist[1]).GetDistance(A)
            BAA2_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[1])
            A3_Atomic_Num[index]=None
            A3_Hybrid[index]=None
            A3_Valence[index]=None
            A3_Aromatic.append(None)
            A3_Distance[index]=None            
            BAA3_Angle[index]=None
            A1AA2_Angle[index]=mol.GetAtom(alist[0]).GetAngle(A,alist[1])
            A1AA3_Angle[index]=None
            A2AA3_Angle[index]=None
        elif len(alist)==3:
            A1_Atomic_Num[index]=mol.GetAtom(alist[0]).GetAtomicNum()
            A1_Hybrid[index]=mol.GetAtom(alist[0]).GetHyb()
            A1_Valence[index]=mol.GetAtom(alist[0]).GetValence()
            A1_Aromatic.append(mol.GetAtom(alist[0]).IsAromatic())
            A1_Distance[index]=mol.GetAtom(alist[0]).GetDistance(A)
            BAA1_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[0])
            A2_Atomic_Num[index]=mol.GetAtom(alist[1]).GetAtomicNum()
            A2_Hybrid[index]=mol.GetAtom(alist[1]).GetHyb()
            A2_Valence[index]=mol.GetAtom(alist[1]).GetValence()
            A2_Aromatic.append(mol.GetAtom(alist[1]).IsAromatic())
            A2_Distance[index]=mol.GetAtom(alist[1]).GetDistance(A)
            BAA2_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[1])
            A3_Atomic_Num[index]=mol.GetAtom(alist[1]).GetAtomicNum()
            A3_Hybrid[index]=mol.GetAtom(alist[2]).GetHyb()
            A3_Valence[index]=mol.GetAtom(alist[2]).GetValence()
            A3_Aromatic.append(mol.GetAtom(alist[2]).IsAromatic())
            A3_Distance[index]=mol.GetAtom(alist[2]).GetDistance(A)
            BAA3_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[2])
            A1AA2_Angle[index]=mol.GetAtom(alist[0]).GetAngle(A,alist[1])
            A1AA3_Angle[index]=mol.GetAtom(alist[0]).GetAngle(A,alist[2])
            A2AA3_Angle[index]=mol.GetAtom(alist[1]).GetAngle(A,alist[2])
        else:
            print('The len of alist is weird: ',len(alist), '.  The index of the weirdo is:' ,index)
            print('alist = ',alist)
            print('A=', A, 'B=', B)
        
    print(len(A3_Distance))
    xtest1JHN=pd.DataFrame({'AB_Distance':AB_Distance, 'A_Hybrid':A_Hybrid, 'A_Valence':A_Valence, 'A_Aromatic':A_Aromatic, \
                         'A_NearestNeighbors':A_NearestNeighbors, 'A1_Atomic_Num':A1_Atomic_Num, 'A1_Hybrid':A1_Hybrid,\
                         'A1_Valence':A1_Valence, 'A1_Aromatic':A1_Aromatic, 'A1_Distance':A1_Distance, 'BAA1_Angle':BAA1_Angle, \
                         'A2_Atomic_Num': A2_Atomic_Num,'A2_Hybrid': A2_Hybrid, 'A2_Valence':A2_Valence, 'A2_Aromatic':A2_Aromatic,\
                         'A2_Distance':A2_Distance, 'BAA2_Angle':BAA2_Angle,'A3_Atomic_Num':A3_Atomic_Num, 'A3_Hybrid':A3_Hybrid, \
                         'A3_Valence':A3_Valence, 'A3_Aromatic':A3_Aromatic, 'A3_Distance':A3_Distance, 'BAA3_Angle':BAA3_Angle}) 

    stop=time.time()
    print('It took', stop-start,'seconds to run through',n1JHN,'1JHN couplings in the dataset')

    print(df1JHN.head())
    qualitative_features=['A1_Atomic_Num', 'A2_Atomic_Num', 'A3_Atomic_Num', 'A_Valence', 'A1_Valence', 'A2_Valence', 'A3_Valence',
                           'A_Hybrid', 'A1_Hybrid', 'A2_Hybrid', 'A3_Hybrid', 'A_Aromatic', 'A1_Aromatic', 'A2_Aromatic', 'A3_Aromatic']
    all_data=df1JHN.append(xtest1JHN)
    all_data=pd.get_dummies(all_data, columns=qualitative_features)
#    xtest1JHN=pd.get_dummies(xtest1JHN, columns=qualitative_features)
#    df1JHN.columns.difference(xtest1JHN.columns)
    df1JHN=all_data.iloc[:len(train1JHN)]
    xtest1JHN=all_data.iloc[len(train1JHN):]
    model_xgb.fit(df1JHN, y1JHN)
    xgb.plot_importance(model_xgb,max_num_features=12)
    plt.show()
 

    pred2=model_xgb.predict(df1JHN)
    fig, ax = plt.subplots()
    ax.scatter(x=y1JHN, y=pred2)
    ax.set(xlabel='Actual 1JHN Values', ylabel='XGBoost Predicted 1JHN Coupling Constants',title='XGBoost Model vs Real')
    plt.ion()

    submission1JHN=pd.DataFrame({'id': test1JHN['id'], 'scalar_coupling_constant': model_xgb.predict(xtest1JHN)})
    #submission1JHN.to_csv('submission1JHN.csv', index=False)
    score = rmsle_cv(model_xgb,df1JHN,y1JHN)
    print("Xgboost 1JHN score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))    

    #plt.bar(range(len(model_xgb.feature_importances_)),model_xgb.feature_importances_)
    
#%%
def Learn2JHC(train2JHC, test2JHC):
    import math
    n2JHC=len(test2JHC)+len(train2JHC)
    start=time.time()

    AB_Distance=np.zeros(len(train2JHC))
    BC_Distance=np.zeros(len(train2JHC))
    AC_Distance=np.zeros(len(train2JHC))
    ABC_Distance=np.zeros(len(train2JHC))
    ABC_Angle=np.zeros(len(train2JHC))
    A_Aromatic=[]
    A_Hybrid=np.zeros(len(train2JHC))
    A_Valence=np.zeros(len(train2JHC))
    ABC_Angle=np.zeros(len(train2JHC))
    B_Atomic_Num=np.zeros(len(train2JHC))
    B_Hybrid=np.zeros(len(train2JHC))
    B_Valence=np.zeros(len(train2JHC))
    B_Aromatic=[]
    lasso=make_pipeline(RobustScaler(),Lasso(alpha=0.05))
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
    
    n_folds = 5

    def rmsle_cv(model, dataset,y):
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dataset)
        rmse= np.log(-cross_val_score(model, dataset, y, scoring="neg_mean_absolute_error", cv = kf))
        return(rmse)
    
    OBConversion=ob.OBConversion()
    OBConversion.SetInFormat("xyz")
    for index in range(0,len(train2JHC)):
        mol=ob.OBMol()
        mol_name=train2JHC.iloc[index]['molecule_name'] +'.xyz'
        OBConversion.ReadFile(mol,mol_name)
        if mol.GetAtom(train2JHC.iloc[index]['atom_index_0'].item()+1).IsCarbon():
            A=train2JHC.iloc[index]['atom_index_0'].item()+1
            C=train2JHC.iloc[index]['atom_index_1'].item()+1
        else:
            A=train2JHC.iloc[index]['atom_index_1'].item()+1
            C=train2JHC.iloc[index]['atom_index_0'].item()+1
        B=find_path2(mol,train2JHC.iloc[index]['atom_index_0'].item(),train2JHC.iloc[index]['atom_index_1'].item())+1
        AB_Distance[index]=mol.GetAtom(A).GetDistance(B)
        AC_Distance[index]=mol.GetAtom(A).GetDistance(C)
        BC_Distance[index]=mol.GetAtom(B).GetDistance(C)
        ABC_Distance[index]=AB_Distance[index]+BC_Distance[index]
        ABC_Angle[index]=np.deg2rad(mol.GetAtom(A).GetAngle(B,C))
        A_Hybrid[index]=mol.GetAtom(A).GetHyb()
        A_Valence[index]=mol.GetAtom(A).GetValence()
        A_Aromatic.append(mol.GetAtom(A).IsAromatic())
        B_Atomic_Num[index]=mol.GetAtom(B).GetAtomicNum()
        B_Hybrid[index]=mol.GetAtom(B).GetHyb()
        B_Valence[index]=mol.GetAtom(B).GetValence()
        B_Aromatic.append(mol.GetAtom(B).IsAromatic())
        if index%100000==0:
            print('2JHC ran', math.floor(index/100000.),'trainig examples in', start-time.time(),'seconds')
        

    df2JHC=pd.DataFrame({'AB_Distance':AB_Distance, 'BC_Distance': BC_Distance, 'AC_Distance': AC_Distance,  'ABC_Distance': ABC_Distance, 
                         'ABC_Angle':ABC_Angle, 'A_Hybrid':A_Hybrid, 'A_Valence':A_Valence, 'A_Aromatic':A_Aromatic, 'B_Atomic_Num': B_Atomic_Num, 
                         'B_Hybrid': B_Hybrid, 'B_Valence': B_Valence, 'B_Aromatic':B_Aromatic})    
                       
    y2JHC=train2JHC['scalar_coupling_constant'].reset_index(drop=True)
    
#    plt.hist(AC_Distance)
#    plt.xlabel('Distance between H and N.  Should be roughly 2 A')
#    plt.ylabel('Counts')
#    plt.ion()
#    
#    plt.hist(ABC_Distance)
#    plt.xlabel('Topological Distance between H and N.  Should be roughly 3 A')
#    plt.ylabel('Counts')
#    plt.ion()
#    
#    plt.hist(ABC_Angle)
#    plt.ylabel('Counts')
#    plt.xlabel('Angle from A to B to C')
#    plt.ion()
#    
#    fig, ax = plt.subplots()
#    ax.scatter(x=ABC_Distance,y=train2JHC['scalar_coupling_constant'])
#    ax.set(xlabel='Bond Distance in Angsroms', ylabel='NH Scalar Coupling Constant',title='2JHC vs Topological Distance')
#    plt.ion()
#    
#    fig, ax = plt.subplots()
#    ax.scatter(x=ABC_Angle, y=train2JHC['scalar_coupling_constant'])
#    ax.set(xlabel='Molecular Angle in Radians', ylabel='NH Scalar Coupling Constant',title='2JHC vs Angle')
#    plt.ion()
    
    
    AB_Distance=np.zeros(len(test2JHC))
    BC_Distance=np.zeros(len(test2JHC))
    AC_Distance=np.zeros(len(test2JHC))
    ABC_Distance=np.zeros(len(test2JHC))
    ABC_Angle=np.zeros(len(test2JHC))
    A_Aromatic=[]
    A_Hybrid=np.zeros(len(test2JHC))
    A_Valence=np.zeros(len(test2JHC))
    ABC_Angle=np.zeros(len(test2JHC))
    B_Atomic_Num=np.zeros(len(test2JHC))
    B_Hybrid=np.zeros(len(test2JHC))
    B_Valence=np.zeros(len(test2JHC))
    B_Aromatic=[]
    
    for index in range(0,len(test2JHC)):
        mol=ob.OBMol()
        mol_name=test2JHC.iloc[index]['molecule_name'] +'.xyz'
        OBConversion.ReadFile(mol, mol_name)
        if mol.GetAtom(test2JHC.iloc[index]['atom_index_0'].item()+1).IsCarbon():
            A=test2JHC.iloc[index]['atom_index_0'].item()+1
            C=test2JHC.iloc[index]['atom_index_1'].item()+1
        else:
            A=test2JHC.iloc[index]['atom_index_1'].item()+1
            C=test2JHC.iloc[index]['atom_index_0'].item()+1
        B=find_path2(mol,test2JHC.iloc[index]['atom_index_0'].item(),test2JHC.iloc[index]['atom_index_1'].item())+1
        AB_Distance[index]=mol.GetAtom(A).GetDistance(B)
        AC_Distance[index]=mol.GetAtom(A).GetDistance(C)
        BC_Distance[index]=mol.GetAtom(B).GetDistance(C)
        ABC_Distance[index]=AB_Distance[index] + BC_Distance[index]
        ABC_Angle[index]=np.deg2rad(mol.GetAtom(A).GetAngle(B,C))
        A_Hybrid[index]=mol.GetAtom(A).GetHyb()
        A_Valence[index]=mol.GetAtom(A).GetValence()
        A_Aromatic.append(mol.GetAtom(A).IsAromatic())
        B_Atomic_Num[index]=mol.GetAtom(B).GetAtomicNum()
        B_Hybrid[index]=mol.GetAtom(B).GetHyb()
        B_Valence[index]=mol.GetAtom(B).GetValence()
        B_Aromatic.append(mol.GetAtom(B).IsAromatic())
        if index%100000==0:
            print('2JHC ran', math.floor(index/100000.),'test examples in', start-time.time(),'seconds')
    xtest2JHC=pd.DataFrame({'AB_Distance':AB_Distance, 'BC_Distance': BC_Distance, 'AC_Distance': AC_Distance,  'ABC_Distance': ABC_Distance, 
                         'ABC_Angle':ABC_Angle, 'A_Hybrid':A_Hybrid, 'A_Valence':A_Valence, 'A_Aromatic':A_Aromatic, 'B_Atomic_Num': B_Atomic_Num, 
                         'B_Hybrid': B_Hybrid, 'B_Valence': B_Valence, 'B_Aromatic':B_Aromatic}) 

    stop=time.time()
    print('It took', stop-start,'seconds to run through',n2JHC,'2JHC couplings in the dataset')
    start=time.time()
    lasso.fit(pd.get_dummies(df2JHC),y2JHC)
    stop=time.time()
    print('It took', stop-start, 'seconds to train the model on', len(train2JHC),'examples')
    pred2JHC=lasso.predict(pd.get_dummies(xtest2JHC))



    y_pred=lasso.predict(df2JHC)
    fig, ax = plt.subplots()
    ax.scatter(x=y2JHC, y=y_pred)
    ax.set(xlabel='Actual 2JHC Values', ylabel='Predicted 2JHC Coupling Constants',title='Model vs Real')
#    plt.ion()
    print(pred2JHC[0:3])
    

#    
    model_xgb.fit(df2JHC.values, y2JHC)
    pred2=model_xgb.predict(df2JHC.values)
    
    fig, ax = plt.subplots()
    ax.scatter(x=y2JHC, y=pred2)
    ax.set(xlabel='Actual 2JHC Values', ylabel='XGBoost Predicted 2JHC Coupling Constants',title='XGBoost Model vs Real')
    plt.ion()
    submission2JHC=pd.DataFrame({'id': test2JHC['id'], 'scalar_coupling_constant': model_xgb.predict(xtest2JHC.values)})
    submission2JHC.to_csv('submission2JHC.csv', index=False)
    score = rmsle_cv(model_xgb,df2JHC.values,y2JHC)
    print("Xgboost 2JHC score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#%%
#Building Model for the 2JHH Couplings
def Learn2JHH(train2JHH, test2JHH):
    
    n2JHH=len(test2JHH)+len(train2JHH)
    start=time.time()

    AB_Distance=np.zeros(len(train2JHH))
    BC_Distance=np.zeros(len(train2JHH))
    AC_Distance=np.zeros(len(train2JHH))
    ABC_Distance=np.zeros(len(train2JHH))
    ABC_Angle=np.zeros(len(train2JHH))
    A_Aromatic=[]
    A_Hybrid=np.zeros(len(train2JHH))
    A_Valence=np.zeros(len(train2JHH))
    ABC_Angle=np.zeros(len(train2JHH))
    B_Atomic_Num=np.zeros(len(train2JHH))
    B_Hybrid=np.zeros(len(train2JHH))
    B_Valence=np.zeros(len(train2JHH))
    B_Aromatic=[]
    lasso=make_pipeline(RobustScaler(),Lasso(alpha=0.05))
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
    
    n_folds = 5

    def rmsle_cv(model, dataset,y):
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dataset)
        rmse= np.log(-cross_val_score(model, dataset, y, scoring="neg_mean_absolute_error", cv = kf))
        return(rmse)
    
    OBConversion=ob.OBConversion()
    OBConversion.SetInFormat("xyz")
    for index in range(0,len(train2JHH)):
        mol=ob.OBMol()
        mol_name=train2JHH.iloc[index]['molecule_name'] +'.xyz'
        OBConversion.ReadFile(mol,mol_name)
        if mol.GetAtom(train2JHH.iloc[index]['atom_index_0'].item()+1).IsNitrogen():
            A=train2JHH.iloc[index]['atom_index_0'].item()+1
            C=train2JHH.iloc[index]['atom_index_1'].item()+1
        else:
            A=train2JHH.iloc[index]['atom_index_1'].item()+1
            C=train2JHH.iloc[index]['atom_index_0'].item()+1
        B=find_path2(mol,train2JHH.iloc[index]['atom_index_0'].item(),train2JHH.iloc[index]['atom_index_1'].item())+1
        AB_Distance[index]=mol.GetAtom(A).GetDistance(B)
        AC_Distance[index]=mol.GetAtom(A).GetDistance(C)
        BC_Distance[index]=mol.GetAtom(B).GetDistance(C)
        ABC_Distance[index]=AB_Distance[index]+BC_Distance[index]
        ABC_Angle[index]=np.deg2rad(mol.GetAtom(A).GetAngle(B,C))
        A_Hybrid[index]=mol.GetAtom(A).GetHyb()
        A_Valence[index]=mol.GetAtom(A).GetValence()
        A_Aromatic.append(mol.GetAtom(A).IsAromatic())
        B_Atomic_Num[index]=mol.GetAtom(B).GetAtomicNum()
        B_Hybrid[index]=mol.GetAtom(B).GetHyb()
        B_Valence[index]=mol.GetAtom(B).GetValence()
        B_Aromatic.append(mol.GetAtom(B).IsAromatic())
        

    df2JHH=pd.DataFrame({'AB_Distance':AB_Distance, 'BC_Distance': BC_Distance, 'AC_Distance': AC_Distance,  'ABC_Distance': ABC_Distance, 
                         'ABC_Angle':ABC_Angle, 'A_Hybrid':A_Hybrid, 'A_Valence':A_Valence, 'A_Aromatic':A_Aromatic, 'B_Atomic_Num': B_Atomic_Num, 
                         'B_Hybrid': B_Hybrid, 'B_Valence': B_Valence, 'B_Aromatic':B_Aromatic})    
                       
    y2JHH=train2JHH['scalar_coupling_constant'].reset_index(drop=True)
    
    plt.hist(AC_Distance)
    plt.xlabel('Distance between H and N.  Should be roughly 2 A')
    plt.ylabel('Counts')
    plt.ion()
    
    plt.hist(ABC_Distance)
    plt.xlabel('Topological Distance between H and N.  Should be roughly 3 A')
    plt.ylabel('Counts')
    plt.ion()
    
    plt.hist(ABC_Angle)
    plt.ylabel('Counts')
    plt.xlabel('Angle from A to B to C')
    plt.ion()
    
    fig, ax = plt.subplots()
    ax.scatter(x=ABC_Distance,y=train2JHH['scalar_coupling_constant'])
    ax.set(xlabel='Bond Distance in Angsroms', ylabel='NH Scalar Coupling Constant',title='2JHH vs Topological Distance')
    plt.ion()
    
    fig, ax = plt.subplots()
    ax.scatter(x=ABC_Angle, y=train2JHH['scalar_coupling_constant'])
    ax.set(xlabel='Molecular Angle in Radians', ylabel='NH Scalar Coupling Constant',title='2JHH vs Angle')
    plt.ion()
    
    
    AB_Distance=np.zeros(len(test2JHH))
    BC_Distance=np.zeros(len(test2JHH))
    AC_Distance=np.zeros(len(test2JHH))
    ABC_Distance=np.zeros(len(test2JHH))
    ABC_Angle=np.zeros(len(test2JHH))
    A_Aromatic=[]
    A_Hybrid=np.zeros(len(test2JHH))
    A_Valence=np.zeros(len(test2JHH))
    ABC_Angle=np.zeros(len(test2JHH))
    B_Atomic_Num=np.zeros(len(test2JHH))
    B_Hybrid=np.zeros(len(test2JHH))
    B_Valence=np.zeros(len(test2JHH))
    B_Aromatic=[]
    
    for index in range(0,len(test2JHH)):
        mol=ob.OBMol()
        mol_name=test2JHH.iloc[index]['molecule_name'] +'.xyz'
        OBConversion.ReadFile(mol, mol_name)
        if mol.GetAtom(test2JHH.iloc[index]['atom_index_0'].item()+1).IsNitrogen():
            A=test2JHH.iloc[index]['atom_index_0'].item()+1
            C=test2JHH.iloc[index]['atom_index_1'].item()+1
        else:
            A=test2JHH.iloc[index]['atom_index_1'].item()+1
            C=test2JHH.iloc[index]['atom_index_0'].item()+1
        B=find_path2(mol,test2JHH.iloc[index]['atom_index_0'].item(),test2JHH.iloc[index]['atom_index_1'].item())+1
        AB_Distance[index]=mol.GetAtom(A).GetDistance(B)
        AC_Distance[index]=mol.GetAtom(A).GetDistance(C)
        BC_Distance[index]=mol.GetAtom(B).GetDistance(C)
        ABC_Distance[index]=AB_Distance[index] + BC_Distance[index]
        ABC_Angle[index]=np.deg2rad(mol.GetAtom(A).GetAngle(B,C))
        A_Hybrid[index]=mol.GetAtom(A).GetHyb()
        A_Valence[index]=mol.GetAtom(A).GetValence()
        A_Aromatic.append(mol.GetAtom(A).IsAromatic())
        B_Atomic_Num[index]=mol.GetAtom(B).GetAtomicNum()
        B_Hybrid[index]=mol.GetAtom(B).GetHyb()
        B_Valence[index]=mol.GetAtom(B).GetValence()
        B_Aromatic.append(mol.GetAtom(B).IsAromatic())
    
    xtest2JHH=pd.DataFrame({'AB_Distance':AB_Distance, 'BC_Distance': BC_Distance, 'AC_Distance': AC_Distance,  'ABC_Distance': ABC_Distance, 
                         'ABC_Angle':ABC_Angle, 'A_Hybrid':A_Hybrid, 'A_Valence':A_Valence, 'A_Aromatic':A_Aromatic, 'B_Atomic_Num': B_Atomic_Num, 
                         'B_Hybrid': B_Hybrid, 'B_Valence': B_Valence, 'B_Aromatic':B_Aromatic}) 

    stop=time.time()
    print('It took', stop-start,'seconds to run through',n2JHH,'2JHH couplings in the dataset')
    start=time.time()
    lasso.fit(pd.get_dummies(df2JHH),y2JHH)
    stop=time.time()
    print('It took', stop-start, 'seconds to train the model on', len(train2JHH),'examples')
    pred2JHH=lasso.predict(pd.get_dummies(xtest2JHH))



    y_pred=lasso.predict(df2JHH)
    fig, ax = plt.subplots()
    ax.scatter(x=y2JHH, y=y_pred)
    ax.set(xlabel='Actual 2JHH Values', ylabel='Predicted 2JHH Coupling Constants',title='Model vs Real')
    plt.ion()
    print(pred2JHH[0:3])
    

    
    model_xgb.fit(df2JHH.values, y2JHH)
    pred2=model_xgb.predict(df2JHH.values)
    
    fig, ax = plt.subplots()
    ax.scatter(x=y2JHH, y=pred2)
    ax.set(xlabel='Actual 2JHH Values', ylabel='XGBoost Predicted 2JHH Coupling Constants',title='XGBoost Model vs Real')
    plt.ion()
    submission2JHH=pd.DataFrame({'id': test2JHH['id'], 'scalar_coupling_constant': model_xgb.predict(xtest2JHH.values)})
    submission2JHH.to_csv('submission2JHH.csv', index=False)
    score = rmsle_cv(model_xgb,df2JHH.values,y2JHH)
    print("Xgboost 2JHH score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#%%
def Learn1JHC(train1JHC, test1JHC):
    n1JHC=len(test1JHC)+len(train1JHC)
    start=time.time()
    from sklearn.model_selection import GridSearchCV
    
    
    AB_Distance=np.zeros(len(train1JHC))
    A_Aromatic=[]
    A_Hybrid=np.zeros(len(train1JHC))
    A_Valence=np.zeros(len(train1JHC))
    A_NearestNeighbors=np.zeros(len(train1JHC))
    A1_Atomic_Num=np.zeros(len(train1JHC))
    A1_Hybrid=np.zeros(len(train1JHC))
    A1_Valence=np.zeros(len(train1JHC))
    A1_Aromatic=[]
    A1_Distance=np.zeros(len(train1JHC))
    BAA1_Angle=np.zeros(len(train1JHC))
    A2_Atomic_Num=np.zeros(len(train1JHC))
    A2_Hybrid=np.zeros(len(train1JHC))
    A2_Valence=np.zeros(len(train1JHC))
    A2_Aromatic=[]
    A2_Distance=np.zeros(len(train1JHC))
    BAA2_Angle=np.zeros(len(train1JHC))
    A3_Atomic_Num=np.zeros(len(train1JHC))
    A3_Hybrid=np.zeros(len(train1JHC))
    A3_Valence=np.zeros(len(train1JHC))
    A3_Aromatic=[]
    A3_Distance=np.zeros(len(train1JHC))
    BAA3_Angle=np.zeros(len(train1JHC))
    A1AA2_Angle=np.zeros(len(train1JHC))
    A1AA3_Angle=np.zeros(len(train1JHC))
    A2AA3_Angle=np.zeros(len(train1JHC))
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
    
    n_folds = 5

    def rmsle_cv(model, dataset,y):
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dataset)
        rmse= np.log(-cross_val_score(model, dataset, y, scoring="neg_mean_absolute_error", cv = kf))
        return(rmse)
    
    OBConversion=ob.OBConversion()
    OBConversion.SetInFormat("xyz")
    for index in range(0,len(train1JHC)):
        mol=ob.OBMol()
        mol_name=train1JHC.iloc[index]['molecule_name'] +'.xyz'
        OBConversion.ReadFile(mol,mol_name)
        if mol.GetAtom(train1JHC.iloc[index]['atom_index_0'].item()+1).IsNitrogen():
            A=train1JHC.iloc[index]['atom_index_0'].item()+1
            B=train1JHC.iloc[index]['atom_index_1'].item()+1
        else:
            A=train1JHC.iloc[index]['atom_index_1'].item()+1
            B=train1JHC.iloc[index]['atom_index_0'].item()+1
        
        AB_Distance[index]=mol.GetAtom(A).GetDistance(B)
        A_Hybrid[index]=mol.GetAtom(A).GetHyb()
        A_Valence[index]=mol.GetAtom(A).GetValence()
        A_Aromatic.append(mol.GetAtom(A).IsAromatic())
        atom_iter=ob.OBAtomAtomIter(mol.GetAtom(A))
        alist=[]
        for a in atom_iter:
            alist.append(a.GetIdx())
        A_NearestNeighbors[index]=len(alist)    
        alist.remove(B)
        if alist ==[]:
            A1_Aromatic.append(None)
            A2_Aromatic.append(None)
            A3_Aromatic.append(None)
        elif len(alist)==1:
            A1_Atomic_Num[index]=mol.GetAtom(alist[0]).GetAtomicNum()
            A1_Hybrid[index]=mol.GetAtom(alist[0]).GetHyb()
            A1_Valence[index]=mol.GetAtom(alist[0]).GetValence()
            A1_Aromatic.append(mol.GetAtom(alist[0]).IsAromatic())
            A1_Distance[index]=mol.GetAtom(alist[0]).GetDistance(A)
            BAA1_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[0])
            A2_Atomic_Num[index]=None
            A2_Hybrid[index]=None
            A2_Valence[index]=None
            A2_Aromatic.append(None)
            A2_Distance[index]=None
            BAA2_Angle[index]=None
            A3_Atomic_Num[index]=None
            A3_Hybrid[index]=None
            A3_Valence[index]=None
            A3_Aromatic.append(None)
            A3_Distance[index]=None
            BAA3_Angle[index]=None
            A1AA2_Angle[index]=None
            A1AA3_Angle[index]=None
            A2AA3_Angle[index]=None
        elif len(alist)==2:
            A1_Atomic_Num[index]=mol.GetAtom(alist[0]).GetAtomicNum()
            A1_Hybrid[index]=mol.GetAtom(alist[0]).GetHyb()
            A1_Valence[index]=mol.GetAtom(alist[0]).GetValence()
            A1_Aromatic.append(mol.GetAtom(alist[0]).IsAromatic())
            A1_Distance[index]=mol.GetAtom(alist[0]).GetDistance(A)
            BAA1_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[0])
            A2_Atomic_Num[index]=mol.GetAtom(alist[1]).GetAtomicNum()
            A2_Hybrid[index]=mol.GetAtom(alist[1]).GetHyb()
            A2_Valence[index]=mol.GetAtom(alist[1]).GetValence()
            A2_Aromatic.append(mol.GetAtom(alist[1]).IsAromatic())
            A2_Distance[index]=mol.GetAtom(alist[1]).GetDistance(A)
            BAA2_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[1])
            A3_Atomic_Num[index]=None
            A3_Hybrid[index]=None
            A3_Valence[index]=None
            A3_Aromatic.append(None)
            A3_Distance[index]=None            
            BAA3_Angle[index]=None
            A1AA2_Angle[index]=mol.GetAtom(alist[0]).GetAngle(A,alist[1])
            A1AA3_Angle[index]=None
            A2AA3_Angle[index]=None
        elif len(alist)==3:
            A1_Atomic_Num[index]=mol.GetAtom(alist[0]).GetAtomicNum()
            A1_Hybrid[index]=mol.GetAtom(alist[0]).GetHyb()
            A1_Valence[index]=mol.GetAtom(alist[0]).GetValence()
            A1_Aromatic.append(mol.GetAtom(alist[0]).IsAromatic())
            A1_Distance[index]=mol.GetAtom(alist[0]).GetDistance(A)
            BAA1_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[0])
            A2_Atomic_Num[index]=mol.GetAtom(alist[1]).GetAtomicNum()
            A2_Hybrid[index]=mol.GetAtom(alist[1]).GetHyb()
            A2_Valence[index]=mol.GetAtom(alist[1]).GetValence()
            A2_Aromatic.append(mol.GetAtom(alist[1]).IsAromatic())
            A2_Distance[index]=mol.GetAtom(alist[1]).GetDistance(A)
            BAA2_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[1])
            A3_Atomic_Num[index]=mol.GetAtom(alist[1]).GetAtomicNum()
            A3_Hybrid[index]=mol.GetAtom(alist[2]).GetHyb()
            A3_Valence[index]=mol.GetAtom(alist[2]).GetValence()
            A3_Aromatic.append(mol.GetAtom(alist[2]).IsAromatic())
            A3_Distance[index]=mol.GetAtom(alist[2]).GetDistance(A)
            BAA3_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[2])
            A1AA2_Angle[index]=mol.GetAtom(alist[0]).GetAngle(A,alist[1])
            A1AA3_Angle[index]=mol.GetAtom(alist[0]).GetAngle(A,alist[2])
            A2AA3_Angle[index]=mol.GetAtom(alist[1]).GetAngle(A,alist[2])
        else:
            print('The len of alist is weird: ',len(alist), '.  The index of the weirdo is:' ,index)
            print('alist = ',alist)
            print('A=', A, 'B=', B)
        if index==0:
            print(A2_Aromatic)
            
    print(len(A1_Aromatic), len(A2_Aromatic), len(A1_Hybrid), len(A_NearestNeighbors),len(A3_Distance)) 

    df1JHC=pd.DataFrame({'AB_Distance':AB_Distance, 'A_Hybrid':A_Hybrid, 'A_Valence':A_Valence, 'A_Aromatic':A_Aromatic, 
                         'A_NearestNeighbors':A_NearestNeighbors, 'A1_Atomic_Num':A1_Atomic_Num, 'A1_Hybrid':A1_Hybrid,
                         'A1_Valence':A1_Valence, 'A1_Aromatic':A1_Aromatic, 'A1_Distance':A1_Distance, 'BAA1_Angle':BAA1_Angle, 
                         'A2_Atomic_Num': A2_Atomic_Num,'A2_Hybrid': A2_Hybrid, 'A2_Valence':A2_Valence, 'A2_Aromatic':A2_Aromatic,
                         'A2_Distance':A2_Distance, 'BAA2_Angle':BAA2_Angle,'A3_Atomic_Num':A3_Atomic_Num, 'A3_Hybrid':A3_Hybrid, 
                         'A3_Valence':A3_Valence, 'A3_Aromatic':A3_Aromatic, 'A3_Distance':A3_Distance, 'BAA3_Angle':BAA3_Angle,
                         'A1AA2_Angle':A1AA2_Angle, 'A1AA3_Angle':A1AA3_Angle, 'A2AA3_Angle':A2AA3_Angle})    
                       
    y1JHC=train1JHC['scalar_coupling_constant'].reset_index(drop=True)
    
    plt.hist(AB_Distance)
    plt.xlabel('Distance between H and N.  Should be roughly 2 A')
    plt.ylabel('Counts')
    plt.ion()
    
   
    
    fig, ax = plt.subplots()
    ax.scatter(x=AB_Distance,y=train1JHC['scalar_coupling_constant'])
    ax.set(xlabel='Bond Distance in Angsroms', ylabel='NH Scalar Coupling Constant',title='1JHC vs Topological Distance')
    plt.ion()
    
   
    
    
    AB_Distance=np.zeros(len(test1JHC))
    A_Aromatic=[]
    A_Hybrid=np.zeros(len(test1JHC))
    A_Valence=np.zeros(len(test1JHC))
    A_NearestNeighbors=np.zeros(len(test1JHC))
    A1_Atomic_Num=np.zeros(len(test1JHC))
    A1_Hybrid=np.zeros(len(test1JHC))
    A1_Valence=np.zeros(len(test1JHC))
    A1_Aromatic=[]
    A1_Distance=np.zeros(len(test1JHC))
    BAA1_Angle=np.zeros(len(test1JHC))
    A2_Atomic_Num=np.zeros(len(test1JHC))
    A2_Hybrid=np.zeros(len(test1JHC))
    A2_Valence=np.zeros(len(test1JHC))
    A2_Aromatic=[]
    A2_Distance=np.zeros(len(test1JHC))
    BAA2_Angle=np.zeros(len(test1JHC))
    A3_Atomic_Num=np.zeros(len(test1JHC))
    A3_Hybrid=np.zeros(len(test1JHC))
    A3_Valence=np.zeros(len(test1JHC))
    A3_Aromatic=[]
    A3_Distance=np.zeros(len(test1JHC))
    BAA3_Angle=np.zeros(len(test1JHC))
    A1AA2_Angle=np.zeros(len(test1JHC))
    A1AA3_Angle=np.zeros(len(test1JHC))
    A2AA3_Angle=np.zeros(len(test1JHC))
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.005, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=22000,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
    
    n_folds = 5

    def rmsle_cv(model, dataset,y):
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dataset)
        rmse= np.log(-cross_val_score(model, dataset, y, scoring="neg_mean_absolute_error", cv = kf))
        return(rmse)
    
    OBConversion=ob.OBConversion()
    OBConversion.SetInFormat("xyz")
    for index in range(0,len(test1JHC)):
        mol=ob.OBMol()
        mol_name=test1JHC.iloc[index]['molecule_name'] +'.xyz'
        OBConversion.ReadFile(mol,mol_name)
        if mol.GetAtom(test1JHC.iloc[index]['atom_index_0'].item()+1).IsNitrogen():
            A=test1JHC.iloc[index]['atom_index_0'].item()+1
            B=test1JHC.iloc[index]['atom_index_1'].item()+1
        else:
            A=test1JHC.iloc[index]['atom_index_1'].item()+1
            B=test1JHC.iloc[index]['atom_index_0'].item()+1
        
        AB_Distance[index]=mol.GetAtom(A).GetDistance(B)
        A_Hybrid[index]=mol.GetAtom(A).GetHyb()
        A_Valence[index]=mol.GetAtom(A).GetValence()
        A_Aromatic.append(mol.GetAtom(A).IsAromatic())
        atom_iter=ob.OBAtomAtomIter(mol.GetAtom(A))
        alist=[]
        for a in atom_iter:
            alist.append(a.GetIdx())
        A_NearestNeighbors[index]=len(alist)    
        alist.remove(B)
        if alist ==[]:
            A1_Aromatic.append(None)
            A2_Aromatic.append(None)
            A3_Aromatic.append(None)
        elif len(alist)==1:
            A1_Atomic_Num[index]=mol.GetAtom(alist[0]).GetAtomicNum()
            A1_Hybrid[index]=mol.GetAtom(alist[0]).GetHyb()
            A1_Valence[index]=mol.GetAtom(alist[0]).GetValence()
            A1_Aromatic.append(mol.GetAtom(alist[0]).IsAromatic())
            A1_Distance[index]=mol.GetAtom(alist[0]).GetDistance(A)
            BAA1_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[0])
            A2_Atomic_Num[index]=None
            A2_Hybrid[index]=None
            A2_Valence[index]=None
            A2_Aromatic.append(None)
            A2_Distance[index]=None
            BAA2_Angle[index]=None
            A3_Atomic_Num[index]=None
            A3_Hybrid[index]=None
            A3_Valence[index]=None
            A3_Aromatic.append(None)
            A3_Distance[index]=None
            BAA3_Angle[index]=None
            A1AA2_Angle[index]=None
            A1AA3_Angle[index]=None
            A2AA3_Angle[index]=None
        elif len(alist)==2:
            A1_Atomic_Num[index]=mol.GetAtom(alist[0]).GetAtomicNum()
            A1_Hybrid[index]=mol.GetAtom(alist[0]).GetHyb()
            A1_Valence[index]=mol.GetAtom(alist[0]).GetValence()
            A1_Aromatic.append(mol.GetAtom(alist[0]).IsAromatic())
            A1_Distance[index]=mol.GetAtom(alist[0]).GetDistance(A)
            BAA1_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[0])
            A2_Atomic_Num[index]=mol.GetAtom(alist[1]).GetAtomicNum()
            A2_Hybrid[index]=mol.GetAtom(alist[1]).GetHyb()
            A2_Valence[index]=mol.GetAtom(alist[1]).GetValence()
            A2_Aromatic.append(mol.GetAtom(alist[1]).IsAromatic())
            A2_Distance[index]=mol.GetAtom(alist[1]).GetDistance(A)
            BAA2_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[1])
            A3_Atomic_Num[index]=None
            A3_Hybrid[index]=None
            A3_Valence[index]=None
            A3_Aromatic.append(None)
            A3_Distance[index]=None            
            BAA3_Angle[index]=None
            A1AA2_Angle[index]=mol.GetAtom(alist[0]).GetAngle(A,alist[1])
            A1AA3_Angle[index]=None
            A2AA3_Angle[index]=None
        elif len(alist)==3:
            A1_Atomic_Num[index]=mol.GetAtom(alist[0]).GetAtomicNum()
            A1_Hybrid[index]=mol.GetAtom(alist[0]).GetHyb()
            A1_Valence[index]=mol.GetAtom(alist[0]).GetValence()
            A1_Aromatic.append(mol.GetAtom(alist[0]).IsAromatic())
            A1_Distance[index]=mol.GetAtom(alist[0]).GetDistance(A)
            BAA1_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[0])
            A2_Atomic_Num[index]=mol.GetAtom(alist[1]).GetAtomicNum()
            A2_Hybrid[index]=mol.GetAtom(alist[1]).GetHyb()
            A2_Valence[index]=mol.GetAtom(alist[1]).GetValence()
            A2_Aromatic.append(mol.GetAtom(alist[1]).IsAromatic())
            A2_Distance[index]=mol.GetAtom(alist[1]).GetDistance(A)
            BAA2_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[1])
            A3_Atomic_Num[index]=mol.GetAtom(alist[1]).GetAtomicNum()
            A3_Hybrid[index]=mol.GetAtom(alist[2]).GetHyb()
            A3_Valence[index]=mol.GetAtom(alist[2]).GetValence()
            A3_Aromatic.append(mol.GetAtom(alist[2]).IsAromatic())
            A3_Distance[index]=mol.GetAtom(alist[2]).GetDistance(A)
            BAA3_Angle[index]=mol.GetAtom(B).GetAngle(A,alist[2])
            A1AA2_Angle[index]=mol.GetAtom(alist[0]).GetAngle(A,alist[1])
            A1AA3_Angle[index]=mol.GetAtom(alist[0]).GetAngle(A,alist[2])
            A2AA3_Angle[index]=mol.GetAtom(alist[1]).GetAngle(A,alist[2])
        else:
            print('The len of alist is weird: ',len(alist), '.  The index of the weirdo is:' ,index)
            print('alist = ',alist)
            print('A=', A, 'B=', B)
        
    print(len(A3_Distance))
    xtest1JHC=pd.DataFrame({'AB_Distance':AB_Distance, 'A_Hybrid':A_Hybrid, 'A_Valence':A_Valence, 'A_Aromatic':A_Aromatic, 
                         'A_NearestNeighbors':A_NearestNeighbors, 'A1_Atomic_Num':A1_Atomic_Num, 'A1_Hybrid':A1_Hybrid,
                         'A1_Valence':A1_Valence, 'A1_Aromatic':A1_Aromatic, 'A1_Distance':A1_Distance, 'BAA1_Angle':BAA1_Angle, 
                         'A2_Atomic_Num': A2_Atomic_Num,'A2_Hybrid': A2_Hybrid, 'A2_Valence':A2_Valence, 'A2_Aromatic':A2_Aromatic,
                         'A2_Distance':A2_Distance, 'BAA2_Angle':BAA2_Angle,'A3_Atomic_Num':A3_Atomic_Num, 'A3_Hybrid':A3_Hybrid, 
                         'A3_Valence':A3_Valence, 'A3_Aromatic':A3_Aromatic, 'A3_Distance':A3_Distance, 'BAA3_Angle':BAA3_Angle}) 

    stop=time.time()
    print('It took', stop-start,'seconds to run through',n1JHC,'1JHC couplings in the dataset')

    print(df1JHC.head())
    qualitative_features=['A1_Atomic_Num', 'A2_Atomic_Num', 'A3_Atomic_Num', 'A_Valence', 'A1_Valence', 'A2_Valence', 'A3_Valence',
                           'A_Hybrid', 'A1_Hybrid', 'A2_Hybrid', 'A3_Hybrid', 'A_Aromatic', 'A1_Aromatic', 'A2_Aromatic', 'A3_Aromatic']
    all_data=df1JHC.append(xtest1JHC)
    all_data=pd.get_dummies(all_data, columns=qualitative_features)
#    xtest1JHC=pd.get_dummies(xtest1JHC, columns=qualitative_features)
#    df1JHC.columns.difference(xtest1JHC.columns)
    df1JHC=all_data.iloc[:len(train1JHC)]
    xtest1JHC=all_data.iloc[len(train1JHC):]
    model_xgb.fit(df1JHC, y1JHC)
    xgb.plot_importance(model_xgb,max_num_features=12)
    plt.show()
 

    pred2=model_xgb.predict(df1JHC)
    fig, ax = plt.subplots()
    ax.scatter(x=y1JHC, y=pred2)
    ax.set(xlabel='Actual 1JHC Values', ylabel='XGBoost Predicted 1JHC Coupling Constants',title='XGBoost Model vs Real')
    plt.ion()

    submission1JHC=pd.DataFrame({'id': test1JHC['id'], 'scalar_coupling_constant': model_xgb.predict(xtest1JHC)})
    #submission1JHC.to_csv('submission1JHC.csv', index=False)
    score = rmsle_cv(model_xgb,df1JHC,y1JHC)
    print("Xgboost 1JHC score: {:.4f} ({:.4f})\n".format(score.mean(), score.std())) 
    
    folds=5
    Early_Stop=50
    Max_Rounds=10000
    

    #plt.bar(range(len(model_xgb.feature_importances_)),model_xgb.feature_importances_)
    

## 3JHC Couplings
#A is the C index+1, B is the connecting atom index+1, C=Connecting atom+1, D=H+1
## 3JHC Couplings
#A is the C index+1, B is the connecting atom index+1, C=Connecting atom+1, D=H+1
def Learn3JHC(train3JHC, test3JHC):
            
    n3JHC=len(test3JHC)+len(train3JHC)
    start=time.time()
    
    AB_Distance=np.zeros(len(train3JHC))
    BC_Distance=np.zeros(len(train3JHC))
    CD_Distance=np.zeros(len(train3JHC))
    AC_Distance=np.zeros(len(train3JHC))
    AD_Distance=np.zeros(len(train3JHC))
    BD_Distance=np.zeros(len(train3JHC))
    ABC_Distance=np.zeros(len(train3JHC))
    BCD_Distance=np.zeros(len(train3JHC))
    ABCD_Distance=np.zeros(len(train3JHC))
    ABC_Angle=np.zeros(len(train3JHC))
    BCD_Angle=np.zeros(len(train3JHC))
    ABCD_Dihedral=np.zeros(len(train3JHC))
    A_Aromatic=[]
    A_Hybrid=np.zeros(len(train3JHC))
    A_Valence=np.zeros(len(train3JHC))
    ABC_Angle=np.zeros(len(train3JHC))
    B_Atomic_Num=np.zeros(len(train3JHC))
    B_Hybrid=np.zeros(len(train3JHC))
    B_Valence=np.zeros(len(train3JHC))
    B_Aromatic=[]
    C_Atomic_Num=np.zeros(len(train3JHC))
    C_Hybrid=np.zeros(len(train3JHC))
    C_Valence=np.zeros(len(train3JHC))
    C_Aromatic=[]
    lasso=make_pipeline(RobustScaler(),Lasso(alpha=0.05))
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
    
    n_folds = 5

    def rmsle_cv(model, dataset,y):
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dataset)
        rmse= np.log(-cross_val_score(model, dataset, y, scoring="neg_mean_absolute_error", cv = kf))
        return(rmse)
    OBConversion=ob.OBConversion()
    OBConversion.SetInFormat("xyz")
    for index in range(0,len(train3JHC)):
        mol_name=train3JHC.iloc[index]['molecule_name'] +'.xyz'
        mol=ob.OBMol()
        OBConversion.ReadFile(mol, mol_name)
        if mol.GetAtom(train3JHC.iloc[index]['atom_index_0'].item()+1).IsCarbon():
            A=train3JHC.iloc[index]['atom_index_0'].item()+1
            D=train3JHC.iloc[index]['atom_index_1'].item()+1
        else:
            A=train3JHC.iloc[index]['atom_index_1'].item()+1
            D=train3JHC.iloc[index]['atom_index_0'].item()+1
        path=find_path3(mol,A-1,D-1)
        B=path[0]+1
        C=path[1]+1
        AB_Distance[index]=mol.GetAtom(A).GetDistance(B)
        AC_Distance[index]=mol.GetAtom(A).GetDistance(C)
        BC_Distance[index]=mol.GetAtom(B).GetDistance(C)
        CD_Distance[index]=mol.GetAtom(C).GetDistance(D)
        AD_Distance[index]=mol.GetAtom(A).GetDistance(D)
        BD_Distance[index]=mol.GetAtom(B).GetDistance(D)
        
        ABC_Distance[index]=AB_Distance[index]+BC_Distance[index]
        BCD_Distance[index]=BC_Distance[index]+CD_Distance[index]
        ABCD_Distance[index]=ABC_Distance[index]+CD_Distance[index]
        ABC_Angle[index]=np.deg2rad(mol.GetAtom(A).GetAngle(B,C))
        BCD_Angle[index]=np.deg2rad(mol.GetAtom(B).GetAngle(C,D))
        ABCD_Dihedral[index]=np.deg2rad(mol.GetTorsion(A,B,C,D))
        A_Hybrid[index]=mol.GetAtom(A).GetHyb()
        A_Valence[index]=mol.GetAtom(A).GetValence()
        A_Aromatic.append(mol.GetAtom(A).IsAromatic())
        B_Atomic_Num[index]=mol.GetAtom(B).GetAtomicNum()
        B_Hybrid[index]=mol.GetAtom(B).GetHyb()
        B_Valence[index]=mol.GetAtom(B).GetValence()
        B_Aromatic.append(mol.GetAtom(B).IsAromatic())
        C_Atomic_Num[index]=mol.GetAtom(C).GetAtomicNum()
        C_Hybrid[index]=mol.GetAtom(C).GetHyb()
        C_Valence[index]=mol.GetAtom(C).GetValence()
        C_Aromatic.append(mol.GetAtom(C).IsAromatic())

        

    df3JHC=pd.DataFrame({'AB_Distance':AB_Distance, 'AC_Distance': AC_Distance, 'AD_Distance': AD_Distance, 'BC_Distance':BC_Distance,
                         'BD_Distance':BD_Distance, 'CD_Distance': CD_Distance, 'ABC_Distance':ABC_Distance, 'BCD_Distance':BCD_Distance,
                         'ABCD_Distance': ABCD_Distance,'ABC_Angle':ABC_Angle, 'BCD_Angle':BCD_Angle, 'ABCD_Dihedral':ABCD_Dihedral, 
                         'A_Hybrid':A_Hybrid, 'A_Valence':A_Valence, 'A_Aromatic':A_Aromatic, 'B_Atomic_Num': B_Atomic_Num, 
                         'B_Hybrid': B_Hybrid, 'B_Valence': B_Valence, 'B_Aromatic':B_Aromatic,'C_Atomic_Num':C_Atomic_Num,'C_Hybrid':C_Hybrid,
                         'C_Valence':C_Valence, 'C_Aromatic':C_Aromatic, 'Cos_Theta':np.cos(ABCD_Dihedral),
                         'Cos_2Theta':np.cos(2*ABCD_Dihedral)})                       
    y3JHC=train3JHC['scalar_coupling_constant'].reset_index(drop=True)
    
    plt.hist(AD_Distance)
    plt.xlabel('Distance between H and N.  Should be roughly 2 A')
    plt.ylabel('Counts')
    plt.ion()
    
    plt.hist(ABCD_Distance)
    plt.xlabel('Topological Distance between H and N.  Should be roughly 3 A')
    plt.ylabel('Counts')
    plt.ion()
    
    plt.hist(ABCD_Dihedral)
    plt.ylabel('Counts')
    plt.xlabel('Dihedral Angle from A to B to C to D')
    plt.ion()
    
    fig, ax = plt.subplots()
    ax.scatter(x=ABCD_Distance,y=train3JHC['scalar_coupling_constant'])
    ax.set(xlabel='Bond Distance in Angsroms', ylabel='NH Scalar Coupling Constant',title='3JHC vs Topological Distance')
    plt.ion()
    
    fig, ax = plt.subplots()
    ax.scatter(x=ABCD_Dihedral, y=train3JHC['scalar_coupling_constant'])
    ax.set(xlabel='Molecular Angle in Radians', ylabel='NH Scalar Coupling Constant',title='3JHC vs Angle')
    plt.ion()


    AB_Distance=np.zeros(len(test3JHC))
    BC_Distance=np.zeros(len(test3JHC))
    CD_Distance=np.zeros(len(test3JHC))
    AC_Distance=np.zeros(len(test3JHC))
    AD_Distance=np.zeros(len(test3JHC))
    BD_Distance=np.zeros(len(test3JHC))
    ABC_Distance=np.zeros(len(test3JHC))
    BCD_Distance=np.zeros(len(test3JHC))
    ABCD_Distance=np.zeros(len(test3JHC))
    ABC_Angle=np.zeros(len(test3JHC))
    BCD_Angle=np.zeros(len(test3JHC))
    ABCD_Dihedral=np.zeros(len(test3JHC))
    A_Aromatic=[]
    A_Hybrid=np.zeros(len(test3JHC))
    A_Valence=np.zeros(len(test3JHC))
    ABC_Angle=np.zeros(len(test3JHC))
    B_Atomic_Num=np.zeros(len(test3JHC))
    B_Hybrid=np.zeros(len(test3JHC))
    B_Valence=np.zeros(len(test3JHC))
    B_Aromatic=[]
    C_Atomic_Num=np.zeros(len(test3JHC))
    C_Hybrid=np.zeros(len(test3JHC))
    C_Valence=np.zeros(len(test3JHC))
    C_Aromatic=[]
    
    
    for index in range(0,len(test3JHC)):
        mol_name=test3JHC.iloc[index]['molecule_name'] +'.xyz'
        mol=ob.OBMol()
        OBConversion.ReadFile(mol,mol_name)
        if mol.GetAtom(test3JHC.iloc[index]['atom_index_0'].item()+1).IsCarbon():
            A=test3JHC.iloc[index]['atom_index_0'].item()+1
            D=test3JHC.iloc[index]['atom_index_1'].item()+1
        else:
            A=test3JHC.iloc[index]['atom_index_1'].item()+1
            D=test3JHC.iloc[index]['atom_index_0'].item()+1
        path=find_path3(mol,A-1,D-1)
        B=path[0]+1
        C=path[1]+1
        AB_Distance[index]=mol.GetAtom(A).GetDistance(B)
        AC_Distance[index]=mol.GetAtom(A).GetDistance(C)
        BC_Distance[index]=mol.GetAtom(B).GetDistance(C)
        CD_Distance[index]=mol.GetAtom(C).GetDistance(D)
        AD_Distance[index]=mol.GetAtom(A).GetDistance(D)
        BD_Distance[index]=mol.GetAtom(B).GetDistance(D)
        
        ABC_Distance[index]=AB_Distance[index]+BC_Distance[index]
        BCD_Distance[index]=BC_Distance[index]+CD_Distance[index]
        ABCD_Distance[index]=ABC_Distance[index]+CD_Distance[index]
        ABC_Angle[index]=np.deg2rad(mol.GetAtom(A).GetAngle(B,C))
        BCD_Angle[index]=np.deg2rad(mol.GetAtom(B).GetAngle(C,D))
        ABCD_Dihedral[index]=np.deg2rad(mol.GetTorsion(A,B,C,D))
        A_Hybrid[index]=mol.GetAtom(A).GetHyb()
        A_Valence[index]=mol.GetAtom(A).GetValence()
        A_Aromatic.append(mol.GetAtom(A).IsAromatic())
        B_Atomic_Num[index]=mol.GetAtom(B).GetAtomicNum()
        B_Hybrid[index]=mol.GetAtom(B).GetHyb()
        B_Valence[index]=mol.GetAtom(B).GetValence()
        B_Aromatic.append(mol.GetAtom(B).IsAromatic())
        C_Atomic_Num[index]=mol.GetAtom(C).GetAtomicNum()
        C_Hybrid[index]=mol.GetAtom(C).GetHyb()
        C_Valence[index]=mol.GetAtom(C).GetValence()
        C_Aromatic.append(mol.GetAtom(C).IsAromatic())
    
            
    
    xtest3JHC=pd.DataFrame({'AB_Distance':AB_Distance, 'AC_Distance': AC_Distance, 'AD_Distance': AD_Distance, 'BC_Distance':BC_Distance,
                         'BD_Distance':BD_Distance, 'CD_Distance': CD_Distance, 'ABC_Distance':ABC_Distance, 'BCD_Distance':BCD_Distance,
                         'ABCD_Distance': ABCD_Distance,'ABC_Angle':ABC_Angle, 'BCD_Angle':BCD_Angle, 'ABCD_Dihedral':ABCD_Dihedral, 
                         'A_Hybrid':A_Hybrid, 'A_Valence':A_Valence, 'A_Aromatic':A_Aromatic, 'B_Atomic_Num': B_Atomic_Num, 
                         'B_Hybrid': B_Hybrid, 'B_Valence': B_Valence, 'B_Aromatic':B_Aromatic,'C_Atomic_Num':C_Atomic_Num,'C_Hybrid':C_Hybrid,
                         'C_Valence':C_Valence, 'C_Aromatic':C_Aromatic, 'Cos_Theta':np.cos(ABCD_Dihedral),
                         'Cos_2Theta':np.cos(2*ABCD_Dihedral)}) 
    
    
    stop=time.time()
    print('It took', stop-start,'seconds to run through',n3JHC,'3JHC couplings in the dataset')
    start=time.time()
    lasso.fit(pd.get_dummies(df3JHC),y3JHC)
    stop=time.time()
    print('It took', stop-start, 'seconds to train the model on', len(train3JHC),'examples')
    pred3JHC=lasso.predict(pd.get_dummies(xtest3JHC))
    
    
    
    y_pred=lasso.predict(df3JHC)
    #fig, ax = plt.subplots()
    #ax.scatter(x=y3JHC, y=y_pred)
    #ax.set(xlabel='Actual 3JHC Values', ylabel='Predicted 3JHC Coupling Constants',title='Model vs Real')
    #plt.ion()
    print(pred3JHC[0:3])
    

    
    start=time.time()
    model_xgb.fit(df3JHC.values, y3JHC)
    pred2=model_xgb.predict(df3JHC.values)
    stop=time.time()
    print('It took', stop-start,'seconds to train xgboost for 3JHC Couplings')
    fig, ax = plt.subplots()
    ax.scatter(x=y3JHC, y=pred2)
    ax.set(xlabel='Actual 3JHC Values', ylabel='XGBoost Predicted 3JHC Coupling Constants',title='XGBoost Model vs Real')
    #plt.ion()
    submission3JHC=pd.DataFrame({'id': test3JHC['id'], 'scalar_coupling_constant': model_xgb.predict(xtest3JHC.values)})
    submission3JHC.to_csv('submission3JHC.csv', index=False)
    score = rmsle_cv(model_xgb,df3JHC.values,y3JHC)
    print("Xgboost 3JHC score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#%%
## 3JHH Couplings
#A is the Nitrogen index+1, B is the connecting atom index+1, C=Hydrogen atom index+1
## 3JHH Couplings
#A is the C index+1, B is the connecting atom index+1, C=Connecting atom+1, D=H+1
def Learn3JHH(train3JHH, test3JHH):
            
    n3JHH=len(test3JHH)+len(train3JHH)
    start=time.time()
    
    AB_Distance=np.zeros(len(train3JHH))
    BC_Distance=np.zeros(len(train3JHH))
    CD_Distance=np.zeros(len(train3JHH))
    AC_Distance=np.zeros(len(train3JHH))
    AD_Distance=np.zeros(len(train3JHH))
    BD_Distance=np.zeros(len(train3JHH))
    ABC_Distance=np.zeros(len(train3JHH))
    BCD_Distance=np.zeros(len(train3JHH))
    ABCD_Distance=np.zeros(len(train3JHH))
    ABC_Angle=np.zeros(len(train3JHH))
    BCD_Angle=np.zeros(len(train3JHH))
    ABCD_Dihedral=np.zeros(len(train3JHH))
    A_Aromatic=[]
    A_Hybrid=np.zeros(len(train3JHH))
    A_Valence=np.zeros(len(train3JHH))
    ABC_Angle=np.zeros(len(train3JHH))
    B_Atomic_Num=np.zeros(len(train3JHH))
    B_Hybrid=np.zeros(len(train3JHH))
    B_Valence=np.zeros(len(train3JHH))
    B_Aromatic=[]
    C_Atomic_Num=np.zeros(len(train3JHH))
    C_Hybrid=np.zeros(len(train3JHH))
    C_Valence=np.zeros(len(train3JHH))
    C_Aromatic=[]
    lasso=make_pipeline(RobustScaler(),Lasso(alpha=0.05))
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
    
    n_folds = 5

    def rmsle_cv(model, dataset,y):
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dataset)
        rmse= np.log(-cross_val_score(model, dataset, y, scoring="neg_mean_absolute_error", cv = kf))
        return(rmse)
    OBConversion=ob.OBConversion()
    OBConversion.SetInFormat("xyz")
    for index in range(0,len(train3JHH)):
        mol_name=train3JHH.iloc[index]['molecule_name'] +'.xyz'
        mol=ob.OBMol()
        OBConversion.ReadFile(mol, mol_name)
        if mol.GetAtom(train3JHH.iloc[index]['atom_index_0'].item()+1).IsNitrogen():
            A=train3JHH.iloc[index]['atom_index_0'].item()+1
            D=train3JHH.iloc[index]['atom_index_1'].item()+1
        else:
            A=train3JHH.iloc[index]['atom_index_1'].item()+1
            D=train3JHH.iloc[index]['atom_index_0'].item()+1
        path=find_path3(mol,A-1,D-1)
        B=path[0]+1
        C=path[1]+1
        AB_Distance[index]=mol.GetAtom(A).GetDistance(B)
        AC_Distance[index]=mol.GetAtom(A).GetDistance(C)
        BC_Distance[index]=mol.GetAtom(B).GetDistance(C)
        CD_Distance[index]=mol.GetAtom(C).GetDistance(D)
        AD_Distance[index]=mol.GetAtom(A).GetDistance(D)
        BD_Distance[index]=mol.GetAtom(B).GetDistance(D)
        
        ABC_Distance[index]=AB_Distance[index]+BC_Distance[index]
        BCD_Distance[index]=BC_Distance[index]+CD_Distance[index]
        ABCD_Distance[index]=ABC_Distance[index]+CD_Distance[index]
        ABC_Angle[index]=np.deg2rad(mol.GetAtom(A).GetAngle(B,C))
        BCD_Angle[index]=np.deg2rad(mol.GetAtom(B).GetAngle(C,D))
        ABCD_Dihedral[index]=np.deg2rad(mol.GetTorsion(A,B,C,D))
        A_Hybrid[index]=mol.GetAtom(A).GetHyb()
        A_Valence[index]=mol.GetAtom(A).GetValence()
        A_Aromatic.append(mol.GetAtom(A).IsAromatic())
        B_Atomic_Num[index]=mol.GetAtom(B).GetAtomicNum()
        B_Hybrid[index]=mol.GetAtom(B).GetHyb()
        B_Valence[index]=mol.GetAtom(B).GetValence()
        B_Aromatic.append(mol.GetAtom(B).IsAromatic())
        C_Atomic_Num[index]=mol.GetAtom(C).GetAtomicNum()
        C_Hybrid[index]=mol.GetAtom(C).GetHyb()
        C_Valence[index]=mol.GetAtom(C).GetValence()
        C_Aromatic.append(mol.GetAtom(C).IsAromatic())

        

    df3JHH=pd.DataFrame({'AB_Distance':AB_Distance, 'AC_Distance': AC_Distance, 'AD_Distance': AD_Distance, 'BC_Distance':BC_Distance,
                         'BD_Distance':BD_Distance, 'CD_Distance': CD_Distance, 'ABC_Distance':ABC_Distance, 'BCD_Distance':BCD_Distance,
                         'ABCD_Distance': ABCD_Distance,'ABC_Angle':ABC_Angle, 'BCD_Angle':BCD_Angle, 'ABCD_Dihedral':ABCD_Dihedral, 
                         'A_Hybrid':A_Hybrid, 'A_Valence':A_Valence, 'A_Aromatic':A_Aromatic, 'B_Atomic_Num': B_Atomic_Num, 
                         'B_Hybrid': B_Hybrid, 'B_Valence': B_Valence, 'B_Aromatic':B_Aromatic,'C_Atomic_Num':C_Atomic_Num,'C_Hybrid':C_Hybrid,
                         'C_Valence':C_Valence, 'C_Aromatic':C_Aromatic, 'Cos_Theta':np.cos(ABCD_Dihedral),
                         'Cos_2Theta':np.cos(2*ABCD_Dihedral)})                       
    y3JHH=train3JHH['scalar_coupling_constant'].reset_index(drop=True)
    
    plt.hist(AD_Distance)
    plt.xlabel('Distance between H and N.  Should be roughly 2 A')
    plt.ylabel('Counts')
    plt.ion()
    
    plt.hist(ABCD_Distance)
    plt.xlabel('Topological Distance between H and N.  Should be roughly 3 A')
    plt.ylabel('Counts')
    plt.ion()
    
    plt.hist(ABCD_Dihedral)
    plt.ylabel('Counts')
    plt.xlabel('Dihedral Angle from A to B to C to D')
    plt.ion()
    
    fig, ax = plt.subplots()
    ax.scatter(x=ABCD_Distance,y=train3JHH['scalar_coupling_constant'])
    ax.set(xlabel='Bond Distance in Angsroms', ylabel='NH Scalar Coupling Constant',title='3JHH vs Topological Distance')
    plt.ion()
    
    fig, ax = plt.subplots()
    ax.scatter(x=ABCD_Dihedral, y=train3JHH['scalar_coupling_constant'])
    ax.set(xlabel='Molecular Angle in Radians', ylabel='NH Scalar Coupling Constant',title='3JHH vs Angle')
    plt.ion()


    AB_Distance=np.zeros(len(test3JHH))
    BC_Distance=np.zeros(len(test3JHH))
    CD_Distance=np.zeros(len(test3JHH))
    AC_Distance=np.zeros(len(test3JHH))
    AD_Distance=np.zeros(len(test3JHH))
    BD_Distance=np.zeros(len(test3JHH))
    ABC_Distance=np.zeros(len(test3JHH))
    BCD_Distance=np.zeros(len(test3JHH))
    ABCD_Distance=np.zeros(len(test3JHH))
    ABC_Angle=np.zeros(len(test3JHH))
    BCD_Angle=np.zeros(len(test3JHH))
    ABCD_Dihedral=np.zeros(len(test3JHH))
    A_Aromatic=[]
    A_Hybrid=np.zeros(len(test3JHH))
    A_Valence=np.zeros(len(test3JHH))
    ABC_Angle=np.zeros(len(test3JHH))
    B_Atomic_Num=np.zeros(len(test3JHH))
    B_Hybrid=np.zeros(len(test3JHH))
    B_Valence=np.zeros(len(test3JHH))
    B_Aromatic=[]
    C_Atomic_Num=np.zeros(len(test3JHH))
    C_Hybrid=np.zeros(len(test3JHH))
    C_Valence=np.zeros(len(test3JHH))
    C_Aromatic=[]
    
    
    for index in range(0,len(test3JHH)):
        mol_name=test3JHH.iloc[index]['molecule_name'] +'.xyz'
        mol=ob.OBMol()
        OBConversion.ReadFile(mol,mol_name)
        if mol.GetAtom(test3JHH.iloc[index]['atom_index_0'].item()+1).IsCarbon():
            A=test3JHH.iloc[index]['atom_index_0'].item()+1
            D=test3JHH.iloc[index]['atom_index_1'].item()+1
        else:
            A=test3JHH.iloc[index]['atom_index_1'].item()+1
            D=test3JHH.iloc[index]['atom_index_0'].item()+1
        path=find_path3(mol,A-1,D-1)
        B=path[0]+1
        C=path[1]+1
        AB_Distance[index]=mol.GetAtom(A).GetDistance(B)
        AC_Distance[index]=mol.GetAtom(A).GetDistance(C)
        BC_Distance[index]=mol.GetAtom(B).GetDistance(C)
        CD_Distance[index]=mol.GetAtom(C).GetDistance(D)
        AD_Distance[index]=mol.GetAtom(A).GetDistance(D)
        BD_Distance[index]=mol.GetAtom(B).GetDistance(D)
        
        ABC_Distance[index]=AB_Distance[index]+BC_Distance[index]
        BCD_Distance[index]=BC_Distance[index]+CD_Distance[index]
        ABCD_Distance[index]=ABC_Distance[index]+CD_Distance[index]
        ABC_Angle[index]=np.deg2rad(mol.GetAtom(A).GetAngle(B,C))
        BCD_Angle[index]=np.deg2rad(mol.GetAtom(B).GetAngle(C,D))
        ABCD_Dihedral[index]=np.deg2rad(mol.GetTorsion(A,B,C,D))
        A_Hybrid[index]=mol.GetAtom(A).GetHyb()
        A_Valence[index]=mol.GetAtom(A).GetValence()
        A_Aromatic.append(mol.GetAtom(A).IsAromatic())
        B_Atomic_Num[index]=mol.GetAtom(B).GetAtomicNum()
        B_Hybrid[index]=mol.GetAtom(B).GetHyb()
        B_Valence[index]=mol.GetAtom(B).GetValence()
        B_Aromatic.append(mol.GetAtom(B).IsAromatic())
        C_Atomic_Num[index]=mol.GetAtom(C).GetAtomicNum()
        C_Hybrid[index]=mol.GetAtom(C).GetHyb()
        C_Valence[index]=mol.GetAtom(C).GetValence()
        C_Aromatic.append(mol.GetAtom(C).IsAromatic())
    
            
    
    xtest3JHH=pd.DataFrame({'AB_Distance':AB_Distance, 'AC_Distance': AC_Distance, 'AD_Distance': AD_Distance, 'BC_Distance':BC_Distance,
                         'BD_Distance':BD_Distance, 'CD_Distance': CD_Distance, 'ABC_Distance':ABC_Distance, 'BCD_Distance':BCD_Distance,
                         'ABCD_Distance': ABCD_Distance,'ABC_Angle':ABC_Angle, 'BCD_Angle':BCD_Angle, 'ABCD_Dihedral':ABCD_Dihedral, 
                         'A_Hybrid':A_Hybrid, 'A_Valence':A_Valence, 'A_Aromatic':A_Aromatic, 'B_Atomic_Num': B_Atomic_Num, 
                         'B_Hybrid': B_Hybrid, 'B_Valence': B_Valence, 'B_Aromatic':B_Aromatic,'C_Atomic_Num':C_Atomic_Num,'C_Hybrid':C_Hybrid,
                         'C_Valence':C_Valence, 'C_Aromatic':C_Aromatic, 'Cos_Theta':np.cos(ABCD_Dihedral),
                         'Cos_2Theta':np.cos(2*ABCD_Dihedral)}) 
    
    
    stop=time.time()
    print('It took', stop-start,'seconds to run through',n3JHH,'3JHH couplings in the dataset')
    start=time.time()
    lasso.fit(pd.get_dummies(df3JHH),y3JHH)
    stop=time.time()
    print('It took', stop-start, 'seconds to train the model on', len(train3JHH),'examples')
    pred3JHH=lasso.predict(pd.get_dummies(xtest3JHH))
    
    
    
    y_pred=lasso.predict(df3JHH)
    fig, ax = plt.subplots()
    ax.scatter(x=y3JHH, y=y_pred)
    ax.set(xlabel='Actual 3JHH Values', ylabel='Predicted 3JHH Coupling Constants',title='Model vs Real')
    plt.ion()
    print(pred3JHH[0:3])
    

    
    start=time.time()
    model_xgb.fit(df3JHH.values, y3JHH)
    pred2=model_xgb.predict(df3JHH.values)
    stop=time.time()
    print('It took', stop-start,'seconds to train xgboost for 3JHH Couplings')
    fig, ax = plt.subplots()
    ax.scatter(x=y3JHH, y=pred2)
    ax.set(xlabel='Actual 3JHH Values', ylabel='XGBoost Predicted 3JHH Coupling Constants',title='XGBoost Model vs Real')
    plt.ion()
    submission3JHH=pd.DataFrame({'id': test3JHH['id'], 'scalar_coupling_constant': model_xgb.predict(xtest3JHH.values)})
    submission3JHH.to_csv('submission3JHH.csv', index=False)

    score = rmsle_cv(model_xgb,df3JHH.values,y3JHH)
    print("Xgboost 3JHH score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#%%

def worker1(train, test, submission2JHN):
    print("ID of process running 1JHN is : {}".format(os.getpid()))
    submission2JHN=Learn1JHN(train,test)
    return submission2JHN

def worker2(train,test, submission2JHN):
    print("ID of process running 1JHC is : {}".format(os.getpid()))
    submission2JHN=Learn1JHC(train, test)
    return submission2JHN

def worker3(train, test, submission2JHN):
    print("ID of process running 2JHN is : {}".format(os.getpid()))
    submission2JHN=Learn2JHN(train,test)
    return submission2JHN

def worker4(train,test, submission2JHN):
    print("ID of process running 2JHC is : {}".format(os.getpid()))
    submission2JHN=Learn2JHC(train, test)
    return submission2JHN

def worker5(train, test, submission2JHN):
    print("ID of process running 2JHH is : {}".format(os.getpid()))
    submission2JHN=Learn2JHH(train,test)
    return submission2JHN

def worker6(train,test, submission2JHN):
    print("ID of process running 3JHN is : {}".format(os.getpid()))
    submission2JHN=Learn3JHN(train, test)
    return submission2JHN

def worker7(train, test, submission2JHN):
    print("ID of process running 3JHC is : {}".format(os.getpid()))
    submission2JHN=Learn3JHC(train,test)
    return submission2JHN

def worker8(train,test, submission2JHN):
    print("ID of process running 3JHH is : {}".format(os.getpid()))
    submission2JHN=Learn3JHH(train, test)
    return submission2JHN




startstart=time.time()
print('hello before we start')
lasso=make_pipeline(RobustScaler(),Lasso(alpha=0.05))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                         learning_rate=0.05, max_depth=3, 
                         min_child_weight=1.7817, n_estimators=2200,
                         reg_alpha=0.4640, reg_lambda=0.8571,
                         subsample=0.5213, silent=1,
                         random_state =7, nthread = -1)
startstart=time.time()
n_folds = 5

def rmsle_cv(model, dataset,y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dataset)
    rmse= np.log(-cross_val_score(model, dataset, y, scoring="neg_mean_absolute_error", cv = kf))
    return(rmse)

os.chdir('C:\\Users\\jjohns\Downloads\champs-scalar-coupling')
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
print(train.head())
fname=train['molecule_name'][0]
print(fname+'.xyz')
os.chdir('structures')
OBConversion=ob.OBConversion()
OBConversion.SetInFormat("xyz")
mols=[]
mols_files=os.listdir()
print('The total number of of molecules is {}'.format(len(mols_files)))
mol=ob.OBMol()
print(mols_files[0])
mols_index=dict(map(reversed, enumerate(mols_files)))
start_time=time.time()

#    index=[]
#    for findex in range(200000):
#        index.append(random.randint(0,130775))
#        f=mols_files[index[-1]]
#        mol=ob.OBMol()
#        OBConversion.ReadFile(mol,f)
#        mols.append(mol)
#    mols_files2=[mols_files[i] for i in index]
#    molecule_dictionary=dict(zip(mols_files2,mols))
#    train=train.iloc[index]
#    test=test.iloc[index]
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
submission2JHN=pd.DataFrame({'id': [], 'scalar_coupling_constant': []})

#p1=mp.Process(target=worker1, args=(train1JHN, test1JHN,  submission2JHN))
#p2=mp.Process(target=worker2, args=(train1JHC, test1JHC,  submission2JHN))
#p3=mp.Process(target=worker3, args=(train2JHN, test2JHN,  submission2JHN))
#p4=mp.Process(target=worker4, args=(train2JHC, test2JHC,  submission2JHN))
#p5=mp.Process(target=worker5, args=(train2JHH, test2JHH,  submission2JHN))
#p6=mp.Process(target=worker6, args=(train3JHN, test3JHN,  submission2JHN))
#p7=mp.Process(target=worker7, args=(train3JHC, test3JHC,  submission2JHN))
#p8=mp.Process(target=worker8, args=(train3JHH, test3JHH,  submission2JHN))
#worker1(train1JHN.sample(frac=0.2, random_state=33), test1JHN, submission2JHN)

#p2.start()
   # p3.start()
#p4.start()    
   # p5.start()
   # p6.start()
#p7.start()
#p8.start()
 
#p2.join()
   # p3.join()
   # p4.join()
   # p5.join()
   # p6.join()
#p7.join()
   # p8.join()

stopstop=time.time()
print('It took', stopstop-startstart, 'seconds to run through 10K dataset.')
print('Estimated time to finish the whole thing is', (stopstop-startstart)/200000.0*4650000.0/3600.0,' hours')
    #%%

#worker1(train1JHN.sample(frac=.33,random_state=33),test1JHN.sample(frac=.01, random_state=33),submission2JHN)
#worker2(train1JHC.sample(n=50000,random_state=33),test1JHC.sample(n=2),submission2JHN)
#worker3(train2JHN.sample(n=50000,random_state=33),test2JHN.sample(n=2),submission2JHN)
#worker4(train2JHC.sample(n=50000,random_state=33), test2JHC.sample(n=2),submission2JHN)
#worker5(train2JHH.sample(n=50000,random_state=33), test2JHH.sample(n=2),submission2JHN)
#worker6(train3JHN.sample(n=50000,random_state=33), test3JHN.sample(n=2),submission2JHN)
#worker7(train3JHC.sample(n=50000, random_state=33),test3JHC.sample(frac=0.002, random_state=33),submission2JHN)   
#worker8(train3JHH.sample(n=50000,random_state=33),test3JHH.sample(n=2),submission2JHN)

    ##Read Input Files
#    submission=pd.DataFrame({'id':[], 'scalar_coupling_constant':[]})
#    submission=submission.append(pd.read_csv('submission1JHC.csv'))
#    submission=submission.append(pd.read_csv('submission1JHN.csv'))
#    submission=submission.append(pd.read_csv('submission2JHC.csv'))
#    submission=submission.append(pd.read_csv('submission2JHH.csv'))
#    submission=submission.append(pd.read_csv('submission2JHN.csv'))
#    submission=submission.append(pd.read_csv('submission3JHC.csv'))
#    submission=submission.append(pd.read_csv('submission3JHH.csv'))
#    submission=submission.append(pd.read_csv('submission3JHN.csv'))
#    submission['id']=submission['id'].astype(int)
#    print('The total files has', len(submission),'entries in it.')
#    submission.to_csv('XGBoost.csv',index=False)
#submission=pd.DataFrame({'id': [], 'scalar_coupling_constant': []})
#submission=submission.append(submission1JHC)
#submission=submission.append(submission1JHN)
#submission=submission.append(submission2JHC)
#submission=submission.append(submission2JHH)
#submission=submission.append(submission2JHN)
#submission=submission.append(submission3JHC)
#submission=submission.append(submission3JHH)
#submission=submission.append(submission3JHN)
#submission['id']=submission['id'].astype(int)
#submission.to_csv('XGBoost.csv',index=False)
import mol_image_3D as mi3D
import tensorflow as tf
import keras
import imp
imp.reload(mi3D)
def Learn1JHN_NN(train1JHN, test1JHN):
      
    n_folds = 5

#    def rmsle_cv(model, dataset,y):
#        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dataset)
#        rmse= np.log(-cross_val_score(model, dataset, y, scoring="neg_mean_absolute_error", cv = kf))
#        return(rmse)
    
    OBConversion=ob.OBConversion()
    OBConversion.SetInFormat("xyz")
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
            X=mi3D.make_image_input(mol,A,B)
        else:
            X=np.append(X,mi3D.make_image_input(mol,A,B),axis=1)
    X=(X-np.mean(X))/(np.std(X))
    Y=np.array(train1JHN['scalar_coupling_constant'].reset_index(drop=True))
    Y=Y.reshape((1,len(train1JHN)))
    print(X.shape)
    model=keras.Sequential([
            keras.layers.Dense(256,activation=tf.nn.relu,input_shape=(64*64*64+6,), kernel_initializer=keras.initializers.he_normal()),
            keras.layers.Dense(128,activation=tf.nn.relu),
            keras.layers.Dense(16,activation=tf.nn.relu),
            keras.layers.Dense(1,activation=tf.nn.relu)])
    model.compile(optimizer='adam', loss='mean_absolute_error',
                  metrics=['mean_absolute_error'])
    history=model.fit(X.T,Y.T,epochs=25, verbose=2)
    plt.plot(history.history['mean_absolute_error'])
    plt.show()
    return history