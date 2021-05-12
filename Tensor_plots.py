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


Created on Mon May 10 16:56:20 2021

@author: albertsmith
"""

import pyDIFRATE as DR
from lipid_selections import sel_res
import numpy as np

ef=DR.frames    #Frames module

assert DR.chimeraX.is_chimera_setup(),'Tensor plots require ChimeraX. Please install ChimeraX first, and then run pyDIFRATE.chimeraX.set_chimera_path(path), with path as the location of the ChimeraX executable'

#%% Load the pdb
top='POPC.pdb'     #Topology (pdb also includes positions at one time point)
mol=DR.molecule(top)     #Load the molecule

res=1
sel_res(mol,res,in_place=True)  #Select bonds corresponding to stored tensors

mol.MDA2pdb(select='resid 1')   #Generate a pdb in the molecule object

"We still need to define the tensor directions to correctly plot the tensors"
mol.tensor_frame(Type='bond',sel1=2,sel2=1)     #Define tensor frames

scale0=[2,2,2.5,2.5]

chimera_cmds=['turn x -90','view all','set bgColor white'] 
A=DR.io.load_DIFRATE('tensors') #Load the tensors

#%% Define the plotting function
def plot_tensor(frame_index,tensor='A_0m_finF',scaling=None):
    """
    Plots the tensors onto the POPC molecule for the requested frame index (0-3).
    By default, we plot the tensors A_0m_finF, that is, the tensor components
    resulting from motion of frame f in frame F. Alternatively, one may set 
    tensor to 'A_0m_PASinF' to obtain tensors results from all motion with frame
    F.
    
    Scaling has be automatically adjusted. Setting scaling will make tensors
    appear larger or smaller (typical values around 1)
    """
    
    A0=A[tensor][frame_index].copy()
    if frame_index==1:
        zero_index=np.concatenate((np.arange(19),[50],[65,66])).astype(int)
        A0[:,zero_index]=0 #Zero out tensors with no motion in this frame
        A0[2,zero_index]=0.0000001 #Take care of some divide-by-zero glitch
    
    scaling=scaling if scaling else scale0[frame_index]    
    DR.chimeraX.draw_tensors(A0,mol,sc=scaling,chimera_cmds=chimera_cmds)