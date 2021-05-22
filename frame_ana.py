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


Created on Fri Apr 30 11:05:59 2021

@author: albertsmith
"""

"""Here, we process and plot the results of the frames analysis, where data is 
initially prepared
"""

import pyDIFRATE as DR
import os
from lipid_selections import avg_equiv,get_labels,get_indices
import matplotlib.pyplot as plt

#%% Load NMR sensitivities
"""The sensitivities for the frame analysis should match those for the NMR analysis.
We make the exception that we only include fast motion for rho0. 
"""
target=DR.io.load_DIFRATE('exper_sens')  #Experimental detector sensitivities
target[0,96:]=0               #We take only slow motions

#%% Load the data
folder='data_frames'
file_fr_avg='f{0}_avg'
frames0=[DR.io.load_DIFRATE(os.path.join(folder,file_fr_avg.format(k))) for k in range(4)]
for f in frames0:f.label=get_labels()[2] #Add labels to this data
#%% Optimize detectors
frames0[0].detect.r_target(target,n=15) #Optimize detectors to match experimental
"Use n=15 to get more accurate reproduction of the experimental sensitivities"

for f in frames0:f.detect=frames0[0].detect #Copy detector object to all frames

frames=[f.fit() for f in frames0]   #Fit all frames

#%% Save results for bon-bon plotting
file_fit='f{0}_fit'
for k,f in enumerate(frames):f.save(os.path.join(folder,file_fit.format(k)))

#%% Average for plotting
frames=[avg_equiv(f) for f in frames] #Average over bonds to the same carbons

#%% Plot results
ylims=[[0,.05],[0,.75],[0,0.75],[0,0.4]]
titles=['One-bond librations',r'Internal chain motion:$\parallel$ to the MOI',\
        r'Internal chain motion:$\perp$ to the MOI'+'\nInternal HG/BB motion',\
        'Chain MOI,\nHG/BB overall motion']

xlabels=[r'$\alpha$',r'$\beta$',r'$\gamma$',r'g$_3$',r'g$_2$',r'g$_1$','1','2',\
        '3\n1','4\n2','5\n3','6\n4','7\n5','8\n6','9\n7','10\n8','11\n9','12\n10',\
        '13\n11','14\n12','15\n13','16\n14','17\n15','18\n16']


ind=get_indices(equiv=True) #Get indices for different part of POPC molecule
nhg=len(ind['hg'])      #Number of data points in HG/BB
nSN1=len(ind['SN1'])    #Number of data points in SN1 chain
nSN2=len(ind['SN2'])    #Number of data points in SN2 chain

xlabels=[r'$\alpha$',r'$\beta$',r'$\gamma$',r'g$_3$',r'g$_2$',r'g$_1$','1','2',\
        '3\n1','4\n2','5\n3','6\n4','7\n5','8\n6','9\n7','10\n8','11\n9','12\n10',\
        '13\n11','14\n12','15\n13','16\n14','17\n15','18\n16']



for f,yl,ttl in zip(frames,ylims,titles):
    "First plot the head group"
    ax=f.plot_rho(index=range(nhg),rho_index=range(4),linestyle='--')       
    for k,a in enumerate(ax):
        "Then plot the Palmitoyl chain"
        clr=a.get_children()[0].get_color()     #Match the colors
        a.plot(range(nhg,nhg+nSN2),f.R[ind['SN2'],k],color=clr,linestyle='-')
        a.plot(range(nhg+2,nhg+2+nSN1),f.R[ind['SN1'],k],color=clr,linestyle=':')
        a.set_xticks(range(nhg+nSN2))
        a.set_ylim(yl)
        if a is ax[-1]:
            a.set_xticklabels(xlabels,rotation=0)
    ax[0].figure.get_children()[1].set_title(ttl)   #Add a title
    ax[0].figure.get_children()[1].set_xlim([-14,-6])   #Add a title
    plt.show()
    
