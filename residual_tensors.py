#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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



Created on Mon May 10 15:56:55 2021

@author: albertsmith
"""

import pyDIFRATE as DR
from lipid_selections import sel_res
import numpy as np
import os

ef=DR.frames    #Frames module

#%% Load the MD trajectory (here use the 256 spins)
top='/Volumes/My Book/MD_256/step6.6_equilibration.gro'     #Topology
pos0='/Volumes/My Book/MD_256/run1.part{1:04d}.xtc'         #Positions

nr=256
pos=[pos0.format(nr,i) for i in range(2,140)]
    
mol=DR.molecule(top,pos)     #Load the molecule

res=1
sel_res(mol,res,in_place=True)

#%% Here we set up the frames    
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

"Loads the data using frames"
if os.path.exists('tensors'):
    A=DR.io.load_DIFRATE('tensors')
else:
    A=ef.frames2tensors(mol=mol,n=10)   #Tensors in paper calculated to higher precision
    DR.io.save_DIFRATE('tensors',A)
"""Output of frames2data is a list of data objects The first is the
directly calculated correlation function, the second is the product of
frames. The remaining are the correlation functions of each individual
motion"""

mol.clear_frames()