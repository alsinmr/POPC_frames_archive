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


Created on Thu Apr 29 15:21:10 2021

@author: albertsmith
"""


"""It is possible to obtain un-optimized detector sensitivities (basically, 
detectors whose sensitivities are equal to the output of SVD). The responses of
these unoptimized detectors may be stored, and then later optimized in a second
step.

This file processes each of 256 copies of POPC (84 correlation functions in
each copy). The correlation functions are analyzed as un-optimized detectors and
stored, with analysis and comparison to experiment coming at a later step. 
"""

import os
import numpy as np
import pyDIFRATE as DR
from lipid_selections import sel_res



#%% Set up for loading MD data
top='/Volumes/My Book/MD_256/step6.6_equilibration.gro'     #Path to the topology file
pos0='/Volumes/My Book/MD_256/run1.part{0:04d}.xtc'         #Path to the xtc (position) files

fi=np.arange(2,140)                 #Indices of the xtc files to use
pos=[pos0.format(k) for k in fi]    #List of all xtc files

mol=DR.molecule(top,pos)         #Load trajectory
"""Mol contains a trajectory object, which allows iteration over the full trajectory.
It also contains attributes sel1 and sel2, which correspond to the list of atoms
that define bonds in the molecule. sel1 and sel2 have attributes "positions", 
which define where an atom is at a given timestep in the trajectory
"""
#%% Now extract correlation functions for each copy of POPC and each bond
folder='data_sim'   #Folder to store the results in 
file='res{0}'  #Filename for results storage


    

nd=20   #Number of unoptimized detectors to store
n=10    #Sparse sampling of trajectory
"""We may sparsely sample the trajectory (reasonable, given we have 1.68x10^6 
time points and 256 copies of POPC). We do this with a log-spacing, where the
log-sampling repeats 10 times across the trajectory. n defines the initial spacing,
so that we take the first 10 time points, and skip the 11th. This defines the
log-spacing, and so we continue to sample more sparsely (repeating this pattern 
10 times!). Log-spacing is motivated by the fact that the difference between say,
the first and second time point may be fairly large in the correlation function,
due to quickly decaying components. However, the difference between the 10000 and
10001 point will not be (it could have some differences due to noise). However,
it cannot have real differences, because all rapidly changing components are 
already gone.
"""

    
r=None      #Clear detectors

for res in range(1,257):    #Sweep over the copies
    if not(os.path.exists(os.path.join(folder,file.format(res)))):  #Don't run this fit if file already exists
        sel_res(mol,res,in_place=True)    #Selects the desired residue 
        "(sets mol.sel1 and mol.sel2 to be lists of the atoms in the bonds for the current POPC molecule)"
        
        data=DR.Ct_fast.Ct2data(mol,n=n)    #Extracts the correlation function, stores in data
        
        if r is None:
            if res==1:                      #If first residue, then we may automatically set the detectors
                data.detect.r_no_opt(nd)    #r_no_opt yields nd un-optimized (direct from SVD) detectors
                r=data.detect
                DR.io.save_DIFRATE(os.path.join(folder,'r'),r)
            else:
                """If res is not None, but r is not defined (calculation has been restarted),
                then we re-load the results from residue 1
                """
                r=DR.io.load_DIFRATE(os.path.join(folder,'r'))
        data.detect=r
                   
        fit=data.fit(save_input=False,parallel=False)    #Perform the fit, clear inputs
        fit.save(os.path.join(folder,file.format(res)))    #Save the results