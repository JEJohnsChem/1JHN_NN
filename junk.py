# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:32:20 2019

@author: jjohns
"""
X=[]
Y=[]
Z=[]
OBConversion=ob.OBConversion()
OBConversion.SetInFormat("xyz")
for index in range(0,len(tset)):
        mol=ob.OBMol()
        mol_name=train1JHN.iloc[index]['molecule_name'] +'.xyz'
        OBConversion.ReadFile(mol,mol_name)
        atom_iter=ob.OBMolAtomIter(mol)
        for a in atom_iter:
            X.append(a.GetX())
            Y.append(a.GetY())
            Z.append(a.GetZ())
            
