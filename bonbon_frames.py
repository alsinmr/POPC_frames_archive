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


Created on Fri Apr 30 16:53:02 2021

@author: albertsmith
"""

"""
Function for plotting the bon-bon plots of a given frame
"""

import pyDIFRATE as DR
import os
import numpy as np
from lipid_selections import sel_res
import time


folder='data_frames'
file='f{0}_fit'

scaling=1/np.array([0.067,0.67,0.67,0.67])
chimera_cmds=['turn x -90','view all','set bgColor white','lighting soft'] 

assert DR.chimeraX.is_chimera_setup(),'Bon-bon plots require ChimeraX. Please install ChimeraX first, and then run pyDIFRATE.chimeraX.set_chimera_path(path), with path as the location of the ChimeraX executable'
for fr in range(4):
    assert os.path.exists(os.path.join(folder,file.format(fr))),"{0} does not exist".format(os.path.join(folder,file.format(fr)))

frame=[DR.io.load_DIFRATE(os.path.join(folder,file.format(fr))) for fr in range(4)]

frame[0].sens.molecule.load_struct('POPC.pdb') #Load a pdb of POPC
for f in frame:f.sens.molecule=frame[0].sens.molecule
frame[0].sens.molecule.MDA2pdb(select='resid 1')
sel_res(frame[0].sens.molecule,1,in_place=True)   #Select correct bonds for plotting

def Bonbon_fr(frame_index,rho_index=None):
    """
    Generates the bon-bon plots for all detectors, for the given frame
    """
    
    if rho_index is None:
        for k in range(4):
            frame[frame_index].draw_rho3D(k,chimera_cmds=chimera_cmds,scaling=scaling[frame_index])
    else:
        frame[frame_index].draw_rho3D(rho_index,chimera_cmds=chimera_cmds,scaling=scaling[frame_index])
    time.sleep(5)   
    """Deletion of frame deletes the pdb of POPC. Delay inserted so 
    that chimera launches before this deletion occurs"""