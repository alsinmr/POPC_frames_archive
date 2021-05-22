#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2021 Albert Smith-Penzel

This file is part of POPC frames archive (PFA).

PFA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PFA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PFA.  If not, see <https://www.gnu.org/licenses/>.


Questions, contact me at:
albert.smith-penzel@medizin.uni-leipzig.de


Created on Tue May 11 12:51:59 2021

@author: albertsmith
"""

import numpy as np

def dist_prod(z,*dist):
    """
    If we have two or more distributions describing individual motions, which
    together make up the total motion, then it is the correlation functions 
    that are multiplied together, NOT the distributions themselves. Then, we 
    need to perform a special operation to get the distribution of correlation
    times resulting from a product of motions. This may be done by calculating
    the effective correlation time resulting from every possible pair of
    correlation times of two distributions. The result may be re-binned into
    a new distribution.
    """
    
    dist=list(dist)
    if len(dist)==2:
        "Just two elements– then we simply get the product of these"
        dist1,dist2=dist
        dz=z[1]-z[0]    #Axis spacing
        
        S21=1-dist1.sum()*dz #Order parameters are just 1-integral_over_distribution
        S22=1-dist2.sum()*dz
        
        "All possible combinations of z for the two distributions"
        z1,z2=[x.reshape(z.size**2) for x in np.meshgrid(z,z)] 
        
        zeff=z1+z2-np.log10(10**z1+10**z2)  #Log-effective correlation time
        "A0 is the amplitude corresponding to all the elements in zeff"
        A0=np.array([x.reshape(z.size**2) for x in np.meshgrid(dist1,dist2)]).prod(0)*dz
        
        i=np.digitize(zeff,z)   #Index mapping all zeff back into the original z index
        
        "Sum up all A for which zeff matches a given bin in z"
        out=np.array([A0[i==i0].sum() for i0 in range(z.size)])
        
        out+=S21*dist2+S22*dist1        #Include contributions from S2 *  
        return out
    else:
        "Run the program recursively over all elements"
        out=dist.pop()
        
        for d in dist:
            out=dist_prod(z,out,d)
        return out