#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:53:01 2019

@author: albertsmith
"""

from scipy.optimize import linprog
import numpy as np

def linprog_par(Y):
    Vt=Y[0]
    k=Y[1]
    ntc=np.shape(Vt)[1]
    try:
        x=linprog(np.sum(Vt,axis=1),-Vt.T,np.zeros(ntc),[Vt[:,k]],1,bounds=(-500,500),method='interior-point',options={'disp' :False})
        x=x['x']
    except:
        x=np.ones(Vt.shape[0])
#    x=linprog(np.sum(Vt,axis=1),-Vt.T,np.zeros(ntc),[Vt[:,k]],1,bounds=(None,None))
#    X[k]=x['x']
    return x