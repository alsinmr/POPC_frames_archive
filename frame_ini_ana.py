#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:28:26 2020

@author: albertsmith
"""

import numpy as np
import os
import pyDIFRATE as DR
from lipid_selections import sel_res

ef=DR.frames #Frames module


nd=20   #Number of detectors for initial save

#%% Load the MD trajectory (here use the 256 spins)
dcd='/Volumes/My Book/MD_256/step6.6_equilibration.gro'     #Path to the topology file
psf0='/Volumes/My Book/MD_256/run1.part{0:04d}.xtc'        #Path to the xtc (position) files
psf=[psf0.format(i) for i in range(2,140)]
mol=DR.molecule(dcd,psf)


nf=4    #Number of frames


folder='data_frames'
file_fr='f{0}_res{1}'
file_ct='ct_res{0}'
file_ctp='ctp_res{0}'

ct=list()   #Store the directly calculated correlation function detector ana
ctp=list()  #Store the product of correlation functions detector ana
frames=list()   #Store detector ana of frames
    

v=None


r=None

for res in range(1,101):
    if os.path.exists(os.path.join(folder,file_fr.format(nf-1,res))):
        frames.append([DR.io.load_DIFRATE(os.path.join(dir0,file_fr.format(k,res))) for k in range(nf)])
        ct.append(DR.io.load_DIFRATE(os.path.join(dir0,file_ct.format(res))))
        ctp.append(DR.io.load_DIFRATE(os.path.join(dir0,file_ctp.format(res))))
        print('reloading residue {0}'.format(res))    
    else:      
        sel_res(mol,res,in_place=True)  
        
        "frames is a list of dictionaries, each dictionary defines a frame"
        
        "Librational frame (defined by selections)"
        frames=[{'Type':'librations','sel1':1,'sel2':2,'full':True}]
        
        "Index for the MOI frames"
        "We omit the headgroup, backbone, and carbonyls"
        "np.nan means this frame won't act on this particular bond"
        frame_index=np.concatenate((np.nan*np.ones(19),0*np.ones(31),[np.nan],1*np.ones(33)))
        "Selections for chain 1, chain 2"
        sel=[mol.sel1[k==frame_index].unique for k in range(2)]
        "Index for MOIxy frame (same as MOI, minus the double bonded carbons)"
        frame_index=np.concatenate((np.nan*np.ones(19),0*np.ones(31),\
                                    [np.nan],1*np.ones(14),[np.nan,np.nan],np.ones(17)))
    #    sel=[mol.sel1[k==frame_index0].unique for k in range(2)]
        
        "Making an index to select the right bonds for the MOIxy frame"
        i1=np.zeros(84)
        i1[np.isnan(frame_index)]=np.nan
        i1[np.logical_not(np.isnan(frame_index))]=np.arange(84-np.sum(np.isnan(frame_index)))
        i2=np.logical_not(np.isnan(frame_index),dtype=bool)
        "Set up the MOIxy frame"
        frames.append({'Type':'MOIxy','sel':sel,'sel1':mol.sel1[i2],\
              'sel2':mol.sel2[i2],'index':frame_index[i2],'frame_index':i1})
        "Set up the MOI frame"
        frame_index=np.concatenate((np.nan*np.ones(19),0*np.ones(31),[np.nan],1*np.ones(33)))
        frames.append({'Type':'MOIz','sel':sel,'frame_index':frame_index})
        
        "Selections for alignment for the Headgroup/BB overall motion"
        sel='name O21 O31 C1 C2 C3 O11'
        frame_index=np.concatenate((np.zeros(19),np.nan*np.ones(31),np.zeros(1),np.nan*np.ones(33)))
        "Superimpose (RMS alignment) frame"
        frames.append({'Type':'superimpose','sel':sel,'resids':res,'frame_index':frame_index})
        
        mol.clear_frames() #Clear existing frames (reseting the selection does NOT delete frames)
        for fr in frames:
            mol.new_frame(**fr) #Load all frames stored in the frame dictionary.
        
        "Define the tensor frame (usually just the bond frame."
        """This frame automatically selects a third reference point, usually a
        bonded carbon. However, we need to swap the selections 1 and 2 for this
        to work properly"""
        mol.tensor_frame(Type='bond',sel1=2,sel2=1)
        
        "Loads the data using frames. Takes first 100000 points of trajectory"
        data=ef.frames2data(mol=mol,tf=100000,n=-1)
        """Output of frames2data is a list of data objects The first is the
        directly calculated correlation function, the second is the product of
        frames. The remaining are the correlation functions of each individual
        motion"""
        
        mol.clear_frames()
        
        if res==1:
            "If this is the first residue, then set r_no_opt, save the result"
            r=data[0].detect
            r.r_no_opt(15)
            DR.io.save_DIFRATE(os.path.join(folder,'r'),r)
        elif r is None:
            "Otherwise, we load r if it's not already in memory"
            r=DR.io.load_DIFRATE(os.path.join(folder,'r'))
        
        for d in data:d.detect=r    #Use same detectors for all analyses
        
        fit0=[d.fit(save_input=False) for d in data]  #Fit all objects in data
        for f in fit0:f.sens.info_in=None #Clear this to make
        
        "Save the results"
        fit0[0].save(os.path.join(folder,file_ct.format(res)))
        fit0[1].save(os.path.join(folder,file_ctp.format(res)))
        for k,f in enumerate(fit0[2:]):
            f.save(os.path.join(folder,file_fr.format(k,res)))
            
        ct.append(fit0[0])
        ctp.append(fit0[1])
        frames.append(fit0[2:])
        
        print(res)

