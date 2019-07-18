# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:18:26 2019

@author: jjohns
"""
import openbabel as ob
import numpy as np
def make_image(mol):
    gshape=64
    epsilon=0.00000001
    grid=np.linspace(-10,10.,gshape)
    XXX,YYY,ZZZ=np.meshgrid(grid,grid,grid)
    
    V=np.zeros((gshape,gshape,gshape))
    atom_iter=ob.OBMolAtomIter(mol)
    for atom in atom_iter:
        atomic_num=atom.GetAtomicNum()
        x=atom.GetX()
        y=atom.GetY()
        z=atom.GetZ()
        RRR=np.sqrt((XXX-x)**2+(YYY-y)**2+(ZZZ-z)**2)
        V=V - atomic_num/(RRR+epsilon)
        
    return V

def make_image_input(mol,A,B,gshape=64):
    """
    Takes an openbabel molecule input, mol, and the atom indices for the two atoms for which  you are interested in the coupling
    returns a flattend numpy array of the electrostatic potential with the X,Y,Z coordinates of A and B appended
    """
    
    epsilon=0.00000001
    grid=np.linspace(-10,10.,gshape)
    XXX,YYY,ZZZ=np.meshgrid(grid,grid,grid)
    
    V=np.zeros((gshape,gshape,gshape))
    atom_iter=ob.OBMolAtomIter(mol)
    for atom in atom_iter:
        atomic_num=atom.GetAtomicNum()
        x=atom.GetX()
        y=atom.GetY()
        z=atom.GetZ()
        RRR=np.sqrt((XXX-x)**2+(YYY-y)**2+(ZZZ-z)**2)
        V=V - atomic_num/(RRR+epsilon)
    V=V.reshape((gshape*gshape*gshape,1))
    output=np.append(V,[mol.GetAtom(A).GetX(), mol.GetAtom(A).GetY(), mol.GetAtom(A).GetZ()])
    output=np.append(output,[mol.GetAtom(B).GetX(), mol.GetAtom(B).GetY(), mol.GetAtom(B).GetZ()])
    output=output.reshape(gshape*gshape*gshape+6,1)
    return output
def make_conv_input(mol,A,B,gshape=64):
    """
    Takes an openbabel molecule input, mol, and the atom indices for the two atoms for which  you are interested in the coupling
    returns a flattend numpy array of the electrostatic potential with the X,Y,Z coordinates of A and B appended
    """
    
    epsilon=0.00000001
    grid=np.linspace(-6.,6.,gshape)
    XXX,YYY,ZZZ=np.meshgrid(grid,grid,grid)
    
    V=np.zeros((gshape,gshape,gshape,2))
    atom_iter=ob.OBMolAtomIter(mol)
    for atom in atom_iter:
        atomic_num=atom.GetAtomicNum()
        x=atom.GetX()
        y=atom.GetY()
        z=atom.GetZ()
        RRR=np.sqrt((XXX-x)**2+(YYY-y)**2+(ZZZ-z)**2)
#        V[:,:,:,0]=V[:,:,:,0] - atomic_num/(RRR+epsilon)
        V[:,:,:,0]=V[:,:,:,0]-atomic_num*np.exp(-0.5*RRR*RRR)
        
    atom=mol.GetAtom(A)
    x=atom.GetX()
    y=atom.GetY()
    z=atom.GetZ()
    RRR2=(XXX-x)**2+(YYY-y)**2+(ZZZ-z)**2
    V[:,:,:,1]=V[:,:,:,1] +np.exp(-0.5*(RRR2))
    atom=mol.GetAtom(B)
    x=atom.GetX()
    y=atom.GetY()
    z=atom.GetZ()
    RRR2=(XXX-x)**2+(YYY-y)**2+(ZZZ-z)**2
    V[:,:,:,1]=V[:,:,:,1] +np.exp(-0.5*(RRR2))
    return V
