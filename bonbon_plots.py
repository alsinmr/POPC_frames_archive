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



Created on Thu Apr 29 16:51:13 2021

@author: albertsmith
"""

"""
Here, we reproduce the plots found in Fig. 2 of our paper, where experimental
and simulated data are combine (experimental wherever we have site-specific
resolution, simulated data elsewhere)
"""

import numpy as np
import pyDIFRATE as DR
from lipid_selections import sel_res

mdfit=DR.io.load_DIFRATE('fit_md')  #Simulated data
expfit=DR.io.load_DIFRATE('fit_exper') #Experimental data


combfit=mdfit.copy()    #We combine the data in this object
combfit.del_exp(range(4,15)) #Delete the experiments that we aren't combining

index=np.concatenate((np.arange(5),[13,14]))    #These are the positions where we use experimental data
for lbl0,R in zip(expfit.label[index],expfit.R[index,:4]):
    for k,lbl in enumerate(combfit.label):
        if lbl0==lbl[:len(lbl0)]:   #See if the beginning of the MD label matches the experiment
            combfit.R[k]=R  #Copy over the data
            
"Here, we create the bon-bon plots"
#List of commands to make nice plots in chimeraX
chimera_cmds=['turn x -90','view all','set bgColor white','lighting soft'] 
scaling=1/combfit.R.max()   #Scaling factor for bon-bon plots (use same for all plots for better comparison)

assert DR.chimeraX.is_chimera_setup(),'Bon-bon plots require ChimeraX. Please install ChimeraX first, and then run pyDIFRATE.chimeraX.set_chimera_path(path), with path as the location of the ChimeraX executable'

combfit.sens.molecule.load_struct('POPC.pdb') #Load a pdb of POPC
sel_res(combfit.sens.molecule,1,in_place=True)   #Select correct bonds for plotting
combfit.sens.molecule.MDA2pdb(select='resid 1') #Transfer pdb into correct location for plotting

def Bonbon(rho_index):
    """
    Plots the detector responses for total motion for the specified detector
    """
    combfit.draw_rho3D(rho_index,scaling=scaling,chimera_cmds=chimera_cmds) #Draw the bon-bon plot