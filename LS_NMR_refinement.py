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


Created on Tue May 11 12:31:48 2021

@author: albertsmith
"""

import pyDIFRATE as DR
from DistProd import dist_prod
from lipid_selections import get_indices,get_ltx_labels
import copy
import numpy as np
import os
import matplotlib.pyplot as plt


#%% Load the experimental results
fit=DR.io.load_DIFRATE('fit_exper')

#%% Load the MD-derived landscapes
dist=DR.io.load_DIFRATE('LS_MD_Fit')
nr=dist[0]['z'].size

"""We need an index to connect resonances that are overlapping experimentally
to the simulated data
"""
ex2sim=[[0],[1],[2],[3],[4],[5],[6,22],[7,23],[8,24],[9,17,25,35,36],\
        [10,15,16,18,26,27,33,34],[11,12,13,14,28],[29,32],[30],[31],\
        [19,37],[20,38],[21,39]]

z0=dist[0]['z0']  #z-axis
dz=z0[1]-z0[0]
#%% Distribution function
def dist_asym(z,sigma,ratio=1): #Skewed Gaussian distribution (sigma different on each side)
    i=np.argmin(np.abs(z0-z))   #Index of the maximum of the distribution
    sigma0=sigma*ratio      #Sigma for the upper half of the distribution
    dist=np.exp(-(z-z0)**2/(2*sigma**2))    #Unskewed distribution
    dist[:i]=np.exp(-(z-z0[:i])**2/(2*sigma0**2)) #Change sigma for first part of distribution
    return dist

#%% Function to scale parameters selected from the total distribution
def scale_par(dist,index,frame_no,par='z',ratio=1,scale=1):
    out=list()
    for fn in frame_no:
        out.append(list())
        A=dist[fn]['A'][index].copy()
        z=dist[fn]['z'][index].copy()
        sigma=dist[fn]['sigma'][index].copy()
        for k,(A1,z1,sigma1) in enumerate(zip(A,z,sigma)):
            "Adjust the selected parameters"
            if par=='z':z1+=scale
            if par=='A':A1*=scale
            if par=='sigma':sigma1*=scale
            "Store the new distribution"
            out[-1].append(dist_asym(z1,sigma1,ratio)*A1)
    "Return new distributions for each frame and position"
    return np.array(out)
        
#%% Function to optimize a fit to experimental data
rhoz=DR.io.load_DIFRATE('exper_sens')   #Experimental sensitivities
rhozCO=DR.io.load_DIFRATE('CSA_exper_sens')
rhozCO=np.concatenate((rhozCO[:1],np.zeros([1,200]),rhozCO[1:]))

def opt_fit(dist,ex_index,frame_no,par='z',ratio=1,scale=None):
    """
    Function to optimize a distribution to better match experimental data. One
    must input dist, which is a list, each element corresponding to a frame,
    where parameters describing the frame are contained within a dictionary.
    
    ex_index determines which experimental data point we optimize
    
    frame_no is one or more frames for which we will optimize a parameter
    
    par is the parameter to be optimized ('z','A','sigma')
    
    scale is an array of multiplication factors which is used to determine the
    range of values over which to sweep the parameter in par (swept relative to
    its initial value)
    (parameter swept over scale*par_ini_value)
    
    Note that distribution is modified IN PLACE. Make sure to take a copy 
    ahead of running if the original data is required.
    """
    frame_no=np.atleast_1d(frame_no) #Which frame number(s) (treat as a 1d array)
    index=np.array(ex2sim[ex_index]) #We average over these simulated positions    

    "Below, we calculate the product of all motions NOT being modified"    
    dist0=list()
    for k,d in enumerate(dist):
        if k not in frame_no:   #Collect all distributions not being modified
            dist0.append(d['dist'][index])
    dist0=np.array(dist0)    
    "Product calculated below"
    new_dist0=np.array([dist_prod(z0,*d) for d in dist0.swapaxes(0,1)])
    
    #%% Setup sweep over parameters to optimize
    if scale is None:
        scale=np.linspace(-.5,.5,50) if par=='z' else np.linspace(.5,2,50)  #How to scale parameters
        
    err=list()
    for s in scale:     #Sweep over all possible scaling parameters
        new_dist=scale_par(dist,index,frame_no,par,ratio,s)
        "Product of the modified distribution with the fixed distributions"
        new_dist=np.array([dist_prod(z0,*[new_dist0[k],*new_dist[:,k]]) for k in range(new_dist0.shape[0])])
        new_dist=new_dist.mean(0) #We just need the average of this distribution
        rho=((rhozCO if 6 in index else rhoz)*new_dist).sum(1)*dz #Calculate rho
        err.append(((rho[:4]-fit.R[ex_index,:4])**2).sum()) #Error vs. experiment (only first 4 detectors)
            
    s=scale[np.argmin(err)] #Best scaling
    
    for fn in frame_no:
        "Move new parameters into dist"
        if par=='z':
            dist[fn][par][index]+=s
        else:
            dist[fn][par][index]*=s
    
    new_dist=scale_par(dist,index,frame_no,par,ratio,0 if par=='z' else 1)     #New distributions
    "Transfer optimized distributions into dist"
    for nd,fn in zip(new_dist,frame_no):
        dist[fn]['dist'][index]=nd       
    "dist modified in-place (no output)"                
    return dist


#%% Here, we run the actual optimization
if os.path.exists('LS_NMR_opt'):
    dist_opt=DR.io.load_DIFRATE('LS_NMR_opt')
else:
    dist_opt=copy.deepcopy(dist)
    for k in range(18):
        dist_opt=opt_fit(dist_opt,k,[1,2],'z',ratio=0.5,scale=np.linspace(-.5,.5,41))  
    DR.io.save_DIFRATE('LS_NMR_opt',dist_opt)
    

#%% Calculate the product of the distributions
prod=np.array([dist_prod(z0,*[d['dist'][k] for d in dist]) for k in range(nr)])
prod_opt=np.array([dist_prod(z0,*[d['dist'][k] for d in dist_opt]) for k in range(nr)])

#%% Calculate the detector responses (averaged over overlapping resonances)
rhoc=np.array([(rhozCO*d).sum(1)*dz if k==6 or k==22 else (rhoz*d).sum(1)*dz for k,d in enumerate(prod)])
rhoc_opt=np.array([(rhozCO*d).sum(1)*dz if k==6 or k==22 else (rhoz*d).sum(1)*dz for k,d in enumerate(prod_opt)])

rhoc=np.array([rhoc[i0].mean(0) for i0 in ex2sim])
rhoc_opt=np.array([rhoc_opt[i0].mean(0) for i0 in ex2sim])

#%% Plot the optimized results
ax=fit.plot_rho(rho_index=range(4),errorbars=True)
for k,a in enumerate(ax):
    a.plot(range(rhoc.shape[0]),rhoc[:,k],color='grey',linestyle='--')
    a.plot(range(rhoc_opt.shape[0]),rhoc_opt[:,k],color='black',linestyle=':')
ax[0].legend(['Sim.','Exp.+Sim.',r'Exper. ($\pm\sigma$)'])

#%% Plot the optimized parameters
ind=get_indices(equiv=True) #Indices for different parts of the molecule
nhg=ind['hg'].size
nSN1=ind['SN1'].size
nSN2=ind['SN2'].size
ylims=[[0,.02],[0,.7],[0,.7],[0,.5]]    #Y-limits for each plot
titles=['One-bond librations',r'Internal chain motion: $\parallel$ to MOI',\
        r'Internal chain motion: $\perp$ to MOI,'+'\nInternal HG/BB motion',\
        'Chain MOI,\nHG/BB overall motion']

ylabels=[r'(1-$S^2$)','FWHM',r'$\tau_{max}$ / ps']
clr=['red','green','blue']
for d,dopt,title in zip(dist[1:3],dist_opt[1:3],titles[1:3]):   #Only re-plot the optimized parameters!
    X=list()
    X.append(d['dist'][:,:-1].sum(1)*dz)
    i=np.array([np.argmin(np.abs(z-z0)) for z in d['z']])   #Indicies of the maximum
    ihl=np.array([np.argmin(np.abs(d['dist'][k,i0::-1]-d['A'][k]*0.5)) for k,i0 in enumerate(i)])
    ihr=np.array([np.argmin(np.abs(d['dist'][k,i0:]-d['A'][k]*0.5)) for k,i0 in enumerate(i)])
    X.append(dz*(ihl+ihr))
    X.append(10**d['z']*1e12)
    
        
    
    Xopt=list()
    Xopt.append(dopt['dist'][:,:-1].sum(1)*dz)
    i=np.array([np.argmin(np.abs(z-z0)) for z in dopt['z']])   #Indicies of the maximum
    ihl=np.array([np.argmin(np.abs(dopt['dist'][k,i0::-1]-dopt['A'][k]*0.5)) for k,i0 in enumerate(i)])
    ihr=np.array([np.argmin(np.abs(dopt['dist'][k,i0:]-dopt['A'][k]*0.5)) for k,i0 in enumerate(i)])
    Xopt.append(dz*(ihl+ihr))
    Xopt.append(10**dopt['z']*1e12)
    
    no_data_index=np.concatenate((np.arange(7),[22,30,31]))
    if d is dist[1]:
        for x in X:x[no_data_index]=np.nan
        for x in Xopt:x[no_data_index]=np.nan
    
    fig=plt.figure()
    ax=[fig.add_subplot(3,1,k) for k in range(1,4)]
    
    for a,c,yl,x,xopt in zip(ax,clr,ylabels,X,Xopt):
        if a==ax[0]:
            a.set_title(title)
        if a==ax[-1]:
            a.semilogy(range(nhg),x[ind['hg']],linestyle='-.',color=c)
            a.semilogy(range(nhg),xopt[ind['hg']],linestyle='-.',color='black')
            
            a.semilogy(range(nhg,nhg+nSN2),x[ind['SN2']],linestyle='-',color=c)
            a.semilogy(range(nhg,nhg+nSN2),xopt[ind['SN2']],linestyle='-',color='black')
            
            a.semilogy(range(nhg+2,nhg+nSN2),x[ind['SN1']],linestyle=':',color=c)
            a.semilogy(range(nhg+2,nhg+nSN2),xopt[ind['SN1']],linestyle=':',color='black')

            a.set_xticks(range(24))
            a.set_xticklabels(get_ltx_labels(),rotation=0)
            
            a.legend(['Sim.','Sim.+Exp.'])
        else:
            a.plot(range(nhg),x[ind['hg']],linestyle='-.',color=c)
            a.plot(range(nhg,nhg+nSN2),x[ind['SN2']],linestyle='-',color=c)
            a.plot(range(nhg+2,nhg+nSN2),x[ind['SN1']],linestyle=':',color=c)
            a.set_xticks(range(24))
            a.set_xticklabels([])
        a.set_ylabel(yl)
    for a in ax[:-1]:
        yl=a.get_ylim()
        a.set_ylim([0,yl[1]])
    
plt.show()  

         