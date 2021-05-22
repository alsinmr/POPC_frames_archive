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



Created on Fri Oct 25 15:50:49 2019

@author: albertsmith
"""

"""
A collection of useful functions for selected particular parts of the POPC 
molecule.
"""
import numpy as np


def sel_res(mol0,res,in_place=False):
    """
    Selects all bonds in a particular POPC, by residue number
    
    mol=sel_res(mol0,res)
    
        or, for in-place selection
    
    sel_res(mol,res,in_place=True)
    
    """
    if np.size(res)>1:
        sel1=mol0.mda_object.atoms[0:0]
        sel2=sel1.copy()
        
        if in_place:
            mol=mol0
        
        for r in res:
            if in_place:
                sel_res(mol,r,in_place=True)
            else:
                mol=sel_res(mol0,r,in_place=False)
            sel1=sel1+mol.sel1
            sel2=sel2+mol.sel2
            
        mol.sel1=sel1
        mol.sel2=sel2
        
        label0=mol.label
        label=list()
        for r in res:
            label.extend(['r{0}_'.format(r)+l for l in label0])
            
        if not(in_place):
            return mol
        
    else:
        if in_place:
            mol=mol0
        else:
            mol=mol0.copy()
        
        sel01,sel02,label0,_,_=get_labels()
        
        a,b=np.unique(sel01,return_inverse=True)
        c,d=np.unique(sel02,return_inverse=True)
        
        uni=mol.mda_object
        sel_all=uni.atoms
        sel0=uni.select_atoms('resid {0}'.format(res))
        
        i1=np.zeros(a.size,'int64')
        for m,s in enumerate(a):
            i1[m]=sel0.select_atoms('name {0}'.format(s)).indices[0]
        mol.sel1=sel_all[i1]
        mol.sel1=mol.sel1[b]
        mol.sel1in=None
        
        i2=np.zeros(c.size,'int64')
        for m,s in enumerate(c):
            i2[m]=sel0.select_atoms('name {0}'.format(s)).indices[0]
        mol.sel2=sel_all[i2]
        mol.sel2=mol.sel2[d]
        mol.sel2in=None
        
        mol.label=np.array(label0)
        
        if not(in_place):
            return mol
    
def avg_equiv(data,in_place=False):
    """
    Averages together equivalent bonds in a lipid
    
    out=avg_equiv(data)
    
        or, for in-place averaging
        
    avg_equiv(data,in_place=True)
    
    """
    if isinstance(data,np.ndarray):
        _,_,_,group,label1=get_labels()
        if data.shape[0]==66:
            group=group[18:]
            group=group-6
            label1=label1[6:]
        elif data.shape[0]==18:
            group=group[:18]
            label1=label1[:6]
        nr=len(label1)
        SZ=np.array(data.shape)
        SZ[0]=nr
        out=np.zeros(SZ)
        for m in np.unique(group).astype(int):
            out[m]=data[group==m].mean(axis=0)
        return out
    else:
        if in_place:
            out=data
        else:
            out=data.copy()
        
    
        R0=data.R
        R0_std=data.R_std
        
        _,_,_,group,label1=get_labels()
        if data.R.shape[0]==66:
            group=group[18:]
            group=group-6
            label1=label1[6:]
        elif data.R.shape[0]==18:
            group=group[:18]
            label1=label1[:6]
        
        nd=R0.shape[1]
        
        Ravg=np.zeros([np.size(label1),nd])
        Rstd=np.zeros(Ravg.shape)
        
        for m in np.unique(group).astype(int):
            Ravg[m]=R0[group==m].mean(axis=0)
            Rstd[m]=np.sqrt(np.square(R0_std[group==m]).mean(axis=0)/np.count_nonzero(group==m))
        
        out.R=Ravg
        out.R_std=Rstd
        out.label=np.array(label1)
        
        if not(in_place):
            return out

def avg_data(data_list,check_sens=True):
    """
    Averages together data with identical sensitivities
    
    out=avg_data(data_list)
    
    where data_list is a list of data objects
    
    """
    
    R=np.zeros(data_list[0].R.shape)
    var=np.zeros(data_list[0].R.shape)
    
    out=data_list[0].copy()
    
    rhoz=out.sens.rhoz()
    for k,d in enumerate(data_list):
#        if check_sens:
#            if np.sqrt(np.square(d.sens.rhoz()-rhoz).sum()/rhoz.size)>1e-4:
#                d=d.copy()
#                d.detect.r_target(rhoz)
#                d=d.fit()
#                print(k)
        R+=d.R
        var+=np.square(d.R_std)
    
    n=np.size(data_list)
    out.R=R/n
    out.R_std=np.sqrt(var/n)
    
    return out
    
def get_labels():
    """
    Returns lists with the various bonds in POPC lipid
    """
    #%% Identification and labeling of bonds for analysis
    sel01=['C13','C13','C13','C14','C14','C14','C15','C15','C15',
           'C12','C12',
           'C11','C11',
           'C1','C1',
           'C2',
           'C3','C3',
           'C31',
           'C32','C32',
           'C33','C33',
           'C34','C34',
           'C35','C35',
           'C36','C36',
           'C37','C37',
           'C38','C38',
           'C39','C39',
           'C310','C310',
           'C311','C311',
           'C312','C312',
           'C313','C313',
           'C314','C314',
           'C315','C315',
           'C316','C316','C316',
           'C21',
           'C22','C22',
           'C23','C23',
           'C24','C24',
           'C25','C25',
           'C26','C26',
           'C27','C27',
           'C28','C28',
           'C29',
           'C210',
           'C211','C211',
           'C212','C212',
           'C213','C213',
           'C214','C214',
           'C215','C215',
           'C216','C216',
           'C217','C217',
           'C218','C218','C218'
           ]
    
    
    sel02=['H13A','H13B','H13C','H14A','H14B','H14C','H15A','H15B','H15C',
           'H12A','H12B',
           'H11A','H11B',
           'HA','HB',
           'HS',
           'HX','HY',
           'O32',
           'H2X','H2Y',
           'H3X','H3Y',
           'H4X','H4Y',
           'H5X','H5Y',
           'H6X','H6Y',
           'H7X','H7Y',
           'H8X','H8Y',
           'H9X','H9Y',
           'H10X','H10Y',
           'H11X','H11Y',
           'H12X','H12Y',
           'H13X','H13Y',
           'H14X','H14Y',
           'H15X','H15Y',
           'H16X','H16Y','H16Z',
           'O22',
           'H2S','H2R',
           'H3S','H3R',
           'H4S','H4R',
           'H5S','H5R',
           'H6S','H6R',
           'H7S','H7R',
           'H8S','H8R',
           'H91',
           'H101',
           'H11S','H11R',
           'H12S','H12R',
           'H13S','H13R',
           'H14S','H14R',
           'H15S','H15R',
           'H16S','H16R',
           'H17S','H17R',
           'H18S','H18R','H18T'
           ]
    
    label0=['gamma1A','gamma1B','gamma1C','gamma2A','gamma2B','gamma2C','gamma3A','gamma3B','gamma3C',
           'betaA','betaB',
           'alphaA','alphaB',
           'g3A','g3B',
           'g2',
           'g1X','g1Y',
           'C1_1',
           'C1_2X','C1_2Y',
           'C1_3X','C1_3Y',
           'C1_4X','C1_4Y',
           'C1_5X','C1_5Y',
           'C1_6X','C1_6Y',
           'C1_7X','C1_7Y',
           'C1_8X','C1_8Y',
           'C1_9X','C1_9Y',
           'C1_10X','C1_10Y',
           'C1_11X','C1_11Y',
           'C1_12X','C1_12Y',
           'C1_13X','C1_13Y',
           'C1_14X','C1_14Y',
           'C1_15X','C1_15Y',
           'C1_16X','C1_16Y','C1_16Z',
           'C2_1',
           'C2_2X','C2_2Y',
           'C2_3X','C2_3Y',
           'C2_4X','C2_4Y',
           'C2_5X','C2_5Y',
           'C2_6X','C2_6Y',
           'C2_7X','C2_7Y',
           'C2_8X','C2_8Y',
           'C2_9',
           'C2_10',
           'C2_11X','C2_11Y',
           'C2_12X','C2_12Y',
           'C2_13X','C2_13Y',
           'C2_14X','C2_14Y',
           'C2_15X','C2_15Y',
           'C2_16X','C2_16Y',
           'C2_17X','C2_17Y',
           'C2_18X','C2_18Y','C2_18Z',
           ]
    "Group indicates equivalent bonds"
    group=np.concatenate(([0,0,0,0,0,0,0,0,0],[1,1],[2,2],[3,3],[4],[5,5],[6],[7,7],[8,8],[9,9],\
                          [10,10],[11,11],[12,12],[13,13],[14,14],[15,15],[16,16],[17,17],\
                          [18,18],[19,19],[20,20],[21,21,21],[22],[23,23],[24,24],[25,25],\
                          [26,26],[27,27],[28,28],[29,29],[30],[31],[32,32],[33,33],\
                          [34,34],[35,35],[36,36],[37,37],[38,38],[39,39,39]))
    
    label1=['gamma','beta','alpha','g3','g2','g1','C1_1','C1_2','C1_3','C1_4',
           'C1_5','C1_6','C1_7','C1_8','C1_9','C1_10','C1_11','C1_12','C1_13',
           'C1_14','C1_15','C1_16','C2_1','C2_2','C2_3','C2_4','C2_5','C2_6',
           'C2_7','C2_8','C2_9','C2_10','C2_11','C2_12','C2_13','C2_14','C2_15',
           'C2_16','C2_17','C2_18']

    return sel01,sel02,label0,group,label1

def get_indices(equiv=False,chain_only=False):
    """
    Indices of the head group, SN1 and SN2 chain. Setting equiv to True
    returns indices if equivalent bonds have been averaged together
    """
    if equiv and chain_only:
        head_group=np.arange(0,dtype=int)
        SN1=np.arange(16,dtype=int)
        SN2=np.arange(16,34,dtype=int)
        chains=np.arange(34,dtype=int)
    elif equiv:
        head_group=np.arange(6,dtype=int)
        SN1=np.arange(6,22,dtype=int)
        SN2=np.arange(22,40,dtype=int)
        chains=np.arange(6,40,dtype=int)
    elif chain_only:
        head_group=np.arange(0,dtype=int)
        SN1=np.arange(32,dtype=int)
        SN2=np.arange(32,66,dtype=int)
        chains=np.arange(66,dtype=int)
    else:
        head_group=np.arange(18,dtype=int)
        SN1=np.arange(18,50,dtype=int)
        SN2=np.arange(50,84,dtype=int)
        chains=np.arange(18,84,dtype=int)
    
    return {'hg':head_group,'SN1':SN1,'SN2':SN2,'chains':chains}

def get_ltx_labels(equiv=True,overlay=False):
    """
    Unfinished: this is the desired behavior for equiv=Ture, overlay=True
    """
    
    labels=[r'$\alpha$',r'$\beta$',r'$\gamma$',r'g$_3$',r'g$_2$',r'g$_1$',\
            '1','2','3\n1','4\n2','5\n3','6\n4','7\n5','8\n6','9\n7','10\n8',\
            '11\n9','12\n10','13\n11','14\n12','15\n13','16\n14','17\n15','18\n16']
    
    return labels
    