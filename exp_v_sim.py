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



Created on Thu Apr 29 16:07:38 2021

@author: albertsmith
"""

import pyDIFRATE as DR
import os
import numpy as np
from lipid_selections import avg_equiv
import matplotlib.pyplot as plt

#%% Load in partially processed simulated data
folder='data_sim'
file='res{0}'


file_avg='avg'

if os.path.exists(os.path.join(folder,file_avg)):        
    #If all 256 files were already averaged, just load the result
    md=DR.io.load_DIFRATE(os.path.join(folder,file_avg.format(256)))
else: 
    md0=list()
    counter=0
    for k in range(1,257):
        if os.path.exists(os.path.join(folder,file.format(k))):
            md0.append(DR.io.load_DIFRATE(os.path.join(folder,file.format(k))))
            counter+=1 #Count up how many files get loaded
    
    md=DR.tools.avg_data(md0)       #Average over all molecule copies
    if counter==256:
        #Save the results if all 256 copies are included
        md.save(os.path.join(folder,file_avg))
#%% Optimize detectors to match experimental detectors
target=DR.io.load_DIFRATE('exper_sens')  #Experimental detector sensitivities
target[0,170:]=0               #We eliminate very slow motions here, that our trajectory cannot reasonably estimate
targetCO=DR.io.load_DIFRATE('CSA_exper_sens')  #Exper. detector sens. for C=O
targetCO[0,170:]=0



"""Here, we match the simulated detectors to the experiment. 
We keep some 'extra' detectors (set n=15), that allow us higher timescale resolution
at the expense of signal to noise"""
md.detect.r_target(target=target,n=15)

mdfit=md.fit()

""" CO data has separate sensitivites. We treat these data points separately,
and then fill the CO analysis back into the original data fit.
"""
mdCO=md.copy() #Copy the data object, pyDIFRATE uses deepcopy by default
"These are all the positions that are NOT C=O (almost all)"
del_index=np.concatenate((np.arange(18),np.arange(19,50),np.arange(51,84)))
mdCO.del_data_pt(del_index) #Delete the non-CO data
mdCO.detect.r_target(target=targetCO,n=15) #Optimize the detectors for C=O

mdfitCO=mdCO.fit()  #Fit the C=O data

"Fill in CO data back into original data object (18 and 50 are positions of CO)"
mdfit.R[18,0]=mdfitCO.R[0,0]        #rho0
mdfit.R[18,2:]=mdfitCO.R[0,1:-1]    #rho2-rho5
mdfit.R[50,0]=mdfitCO.R[1,0]        #same as above, for second C0
mdfit.R[50,2:]=mdfitCO.R[1,1:-1]
mdfit.R_std[18,0]=mdfitCO.R_std[0,0]    
mdfit.R_std[18,2:]=mdfitCO.R_std[0,1:-1]
mdfit.R_std[50,0]=mdfitCO.R_std[1,0]    
mdfit.R_std[50,2:]=mdfitCO.R_std[1,1:-1]
mdfit.R[18,1]=0     #Set rho1 to 0
mdfit.R[50,1]=0
mdfit.R_std[18,1]=0
mdfit.R_std[50,1]=0

mdfit.save('fit_md')    #We need this later for making 3D plots

"Average over equivalent H???C bonds"
mdfita=avg_equiv(mdfit)

"""Average over overlapping resonances
Each element of the outer list contains a list of all positions in the simulation
which overlap to yield the corresponding experimental peak.
"""
ex2sim=[[0],[1],[2],[3],[4],[5],[6,22],[7,23],[8,24],[9,17,25,35,36],\
        [10,15,16,17,18,26,27,33,34],[11,12,13,14,28],[29,32],[30],[31],\
        [19,37],[20,38],[21,39]]

md_avg=mdfita.copy()
R=np.zeros([len(ex2sim),md_avg.R.shape[1]])
R_std=np.zeros([len(ex2sim),md_avg.R.shape[1]])
for k,i in enumerate(ex2sim):
    R[k]=mdfita.R[i].mean(axis=0)
    R_std[k]=mdfita.R_std[i].mean(axis=0)
    
md_avg.R=R
md_avg.R_std=R_std


#%% Plot the results
fit=DR.io.load_DIFRATE('fit_exper') #Load the experimental data

ax=fit.plot_rho(errorbars=True)     #Plot the experimental data
ax0=ax[0].figure.get_children()[1]  #Get the axis for the sensitivity plot
md_avg.sens.plot_rhoz(rho_index=range(5),ax=ax0,color='grey') #Plot the MD-derived sensitivity over the experimental sensitivity

ylims=[[0,1],[0,.65],[0,.65],[0,.65],[0,.0012],[0,.0012]]   #Y-limits to make the plots nicer

x0=np.arange(md_avg.R.shape[0])     #x-axis
for k,(a,ylim) in enumerate(zip(ax,ylims)):     #Loop over the detectors
    if k<4:     #We only compare the sub-microsecond motions
        y0=md_avg.R[:,k]
        a.plot(x0,y0,color='black')
    a.set_ylim(ylim)
    
plt.show()    
