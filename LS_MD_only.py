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



Created on Tue May 11 09:18:25 2021

@author: albertsmith
"""

import pyDIFRATE as DR
from lipid_selections import avg_equiv,get_indices,get_ltx_labels
import numpy as np
import matplotlib.pyplot as plt
import os

"""
Here, we fit detector responses to a 3-parameter distribution (correlation time,
distribution width, and amplitude). However, to generate the fits, we use a 
higher resolution description of the MD-derived dynamics than is used for in-text
figures (Fig. 3). This requires re-optimization of the detector sensitivities at
the first step.

Then, we perform a grid search over correlation time and distribution width. At
each element of the grid, we perform a linear fit of the amplitude. We fit all
but the last detector (the last detector collects contributions to the order
parameter, S2, and so it is not included in the linear fit. It is included,
however, when assessing the quality of the linear fit). After performing linear
fits over all elements of the grid, we search for the best fit for each position
in the molecule.

Finally, we reconstruct the total distribution of motion as the product of
correlation functions resulting from the individual fits.
"""

#%% Load the frames data
folder='data_frames'
file='f{0}_avg'
frames=[DR.io.load_DIFRATE(os.path.join(folder,file.format(fr))) for fr in range(4)]

#%% Fit to new set of detectors
r=frames[0].detect
r.r_auto(9)
for f in frames:f.detect=r
frames=[avg_equiv(f).fit() for f in frames]  #Average equivalent sites, fit results
z0=frames[0].sens.z()   #z-axis (log correlation time)
dz=z0[1]-z0[0]

if os.path.exists('LS_MD_fit'):
    "Landscapes already fit. Just load the results."
    dist=DR.io.load_DIFRATE('LS_MD_fit')
else:
    #%% Generate the distribution functions
    def dist_asym(z,sigma,ratio=1): #Skewed Gaussian distribution (sigma different on each side)
        i=np.argmin(np.abs(z0-z))   #Index of the maximum of the distribution
        sigma0=sigma*ratio      #Sigma for the upper half of the distribution
        dist=np.exp(-(z-z0)**2/(2*sigma**2))    #Unskewed distribution
        dist[:i]=np.exp(-(z-z0[:i])**2/(2*sigma0**2)) #Change sigma for first part of distribution
        return dist
    #%% Generate the fitting functions
    def fit2dist(data,zlim=[-12,-8],sigma_lim=[.1,3],ratio=1):
        """
        Fits detector responses stored in data  to a distribution. We use a 
        simple grid search over correlation time and breadth of distribution,
        combined with a linear fit of the amplitude at each point in the grid
        """
        
        rhoz=data.sens.rhoz()   #Sensitivities of th detectors
        rho=data.R #Detector responses to be fitted
        
        #Values of sigma, z to sweep over
        sigma=np.array([sigma_lim[0]]) if sigma_lim[0]==sigma_lim[1] else \
            np.linspace(sigma_lim[0],sigma_lim[1],100)
        z=np.array(zlim[0]) if zlim[0]==zlim[1] else np.linspace(zlim[0],zlim[1],100)
        
        "Make 1D vectors with all combinations of z,sigma to test"
        z,sigma=[x.reshape(np.prod(x.shape)) for x in np.meshgrid(z,sigma)]
        
        err=list()  #Store the error at each grid element
        A=list()    #Store the fitted amplitude at each grid element
        
        for k,(z1,s1) in enumerate(zip(z,sigma)):
            "Sweep over all values of z,sigma. Fit ALL values in data.R simultaneously"
            dist=dist_asym(z1,s1,ratio) #Unscaled distribution at this grid element
            """The shape of the distribution (no adjustment of the amplitude)
            determines the ratio of the detector responses. Then, these can
            be fit to the responses stored in data by a simple scaling factor (A)
            
            Note, however, that for this fit, we use the 
            """
            rho0=(rhoz*dist*dz).sum(1) #Unscaled detector response
            X=1/((rho0[:-1]**2).sum())*rho0[:-1]
            A.append((X*rho[:,:-1]).sum(1))     #Fitted amplitudes
            rhoc=np.dot(np.atleast_2d(A[-1]).T,np.atleast_2d(rho0)) #Fitted detector responses
            rhoc[:,-1]+=1-A[-1]*dist.sum()*dz  #Contributions to last detector from S2 (S2=1-(A*dist0).sum())
            err.append(((rho-rhoc)**2).sum(1))  #Error at this grid element (values for every position)
            
        err=np.array(err)
        i=np.argmin(err,axis=0) #Indices of best fit for each position
        
        "Select best fits for every position"
        err_fit=np.array([err[i0,k] for k,i0 in enumerate(i)])  #Error of best fit
        Afit=np.array([np.array(A)[i0,k] for k,i0 in enumerate(i)]) #Best fit of amplitudes
        zfit=z[i]  #Best fit correlation times
        sigma_fit=sigma[i]  #Best fit widths
        
        "Re-calculate distributions, detector responses corresponding to the fit results"
        rho_c=list()
        dist=list()
        for A1,z1,s1 in zip(Afit,zfit,sigma_fit):
            dist.append(A1*dist_asym(z1,s1,ratio))  #Fitted distributions
            dist[-1][-1]=1/dz-dist[-1].sum()    #Add unused amplitudes to longest correlation time
            rho_c.append((rhoz*dist[-1]*dz).sum(1)) #Fitted detector responses
        
        "Return dictionary of the results"
        return {'z0':z0,'z':zfit,'sigma':sigma_fit,'A':Afit,'dist':np.array(dist),'rho_c':np.array(rho_c),'err':err_fit}
    

    #%% Begin fitting
    zlims=[[-13.0000,-13],[-13,-6],[-13,-6],[-13,-6]]   #Limits for fitting (one set for each of 4 frames)
    slims=[[.4,.4],[.1,3],[.1,3],[.1,3]]
    ratio=[.5,.5,.5,1]
    
    "Perform the full fit"
    dist=[fit2dist(d,zl,sl,rat) for d,zl,sl,rat in zip(frames,zlims,slims,ratio)] #List of the fit results
    """There is some cross-correlation between fit terms A and sigma, 
    leading to apparent noise in the distribution. However, we may reasonably
    assume that variation of A and sigma is smooth across the molecule. Therefore,
    we regularize sigma (resulting also in smoothing of A), although we insert
    breaks in the regularization corresponding to structural distcontinuity in 
    the POPC molecule"""
    
    def smooth(x,l,breaks=None,order=2):
        """
        Minimizes a derivative of x. One may list breaks, which are indices
        for which smoothing is not performed. l is the regularization parameter.
        order is the derivative to be minimized.
        
        X=smooth(x,l,breaks=None,order=2)
        """
        
        n=x.size
        mat00=np.diag(np.ones(n),k=0)-np.diag(np.ones(n-1),k=1)    
        mat0=np.eye(n)    
        for k in range(order):
            mat0=np.dot(mat00,mat0)
            mat0[-k-1]=0
        if breaks is not None:      #Remove smoothing at each break
            breaks=np.atleast_1d(breaks)
            for b in breaks:
                mat0[b-order:b]=np.zeros(n)
        
        """Target function is to retain the initial parameter values (np.eye(n)), 
        weighted against smoothing the derivative (l*mat0). l determines the
        relative weighting.
        """
        mat=np.concatenate((np.eye(n),l*mat0),axis=0)
        target=np.concatenate((x,np.zeros(n)),axis=0)
        return np.linalg.lstsq(mat,target)[0]
    
    "List of breaks for each frame"
    breaks=[np.array([6]),np.array([*np.arange(8),22,23,30,32]),\
            np.array([*np.arange(8),22,23,30,32]),np.array([6,7,21,22])]
    l=1 #Regularization constant
    
    sigma=[smooth(d['sigma'],l,br,order=1) for d,br in zip(dist,breaks)]
    
    """Now we need to re-fit, with the fixed sigma values. We need a new fit
    function to achieve this"""
    def fitA_z(data,sigma,ratio=1):
        """Fits A and z for a given sigma value (and ratio, for assymetric 
        distributions)
        """
        rhoz=data.sens.rhoz()   #Sensitivities of detectors
        rho=data.R  #Detector responses to be fitted
        
        zswp=np.linspace(-13,-6,100)    #Values to be swept over (now fixed)
        
        "Arrays to store the fit results"
        A=list()
        err=list()
        rho_c=list()
        dist=list()
        zout=list()
        
        for R,s1 in zip(rho,sigma):     #Sweep over all positions
            err0=list()     #Temporary storage for sweep over correlation time
            A0=list()
            for z1 in zswp: #Sweep over correlation time
                dist0=dist_asym(z1,s1,ratio)     #Unscaled distribution
                rho0=(rhoz*dist0*dz).sum(1)     #Unscaled detector responses
                X=1/((rho0[:-1]**2).sum())*rho0[:-1]    
                A0.append((X*R[:-1]).sum()) #Fitted amplitudes
                rhoc0=A0[-1]*rho0           #Fitted detector responses
                rhoc0[-1]+=1-A0[-1]*dist0.sum()*dz  #Correct final detector response (S2 contributions)
                err0.append(((R-rhoc0)**2).sum())   #Calculate error
            i=np.argmin(err0)               #Find index of best-fitted correlation time
            zout.append(zswp[i])            #Fitted correlation time
            A.append(A0[i])                 #Fitted amplitude
            err.append(err0[i])             #Fit error
            dist.append(A[-1]*dist_asym(zout[-1],s1,ratio))    #Fitted distribution
            dist[-1][-1]=1/dz-dist[-1].sum()
            rho_c.append(((rhoz*dist[-1]*dz).sum(1)))   #Fitted detector responses
            
        return {'z0':z0,'z':np.array(zout),'sigma':np.array(sigma),'A':np.array(A),\
                'dist':np.array(dist),'rho_c':np.array(rho_c),'err':np.array(err)}
    
    "Here, we refit the results with the smoothed values of sigma (values are fixed)"    
    dist=[fitA_z(f,s,rat) for f,s,rat in zip(frames,sigma,ratio)]
    
    DR.io.save_DIFRATE('LS_MD_fit',dist)
    
#%% Plot the fit results
ind=get_indices(equiv=True) #Indices for different parts of the molecule
nhg=ind['hg'].size
nSN1=ind['SN1'].size
nSN2=ind['SN2'].size
ylims=[[0,.02],[0,.7],[0,.7],[0,.5]]    #Y-limits for each plot
titles=['One-bond librations',r'Internal chain motion: $\parallel$ to MOI',\
        r'Internal chain motion: $\perp$ to MOI,'+'\nInternal HG/BB motion',\
        'Chain MOI,\nHG/BB overall motion']

for f,d,yl,title in zip(frames,dist,ylims,titles):    #Figure for each frame
    rho_c=d['rho_c']
    ax=f.plot_rho(index=ind['hg'],linestyle='-.') #Creat figure, plot head group
    ax[0].figure.get_children()[1].set_title(title)
    clr=[a.get_children()[0].get_color() for a in ax]
    for k,a in enumerate(ax):   #Loop over each detector
        a.plot(range(nhg),rho_c[ind['hg'],k],linestyle='-.',color='black')   #Fit of head group
        a.plot(range(nhg,nhg+nSN2),f.R[ind['SN2'],k],linestyle='-',color=clr[k]) #SN2 chain
        a.plot(range(nhg,nhg+nSN2),rho_c[ind['SN2'],k],linestyle='-',color='black') #SN2 chain fit
        a.plot(range(nhg+2,nhg+nSN2),f.R[ind['SN1'],k],linestyle=':',color=clr[k]) #SN1 chain
        a.plot(range(nhg+2,nhg+nSN2),rho_c[ind['SN1'],k],linestyle=':',color='black') #SN1 chain fit
        a.set_xticks(range(24))
        if a==ax[-1]:
            a.set_ylim([0,np.max([rho_c[:,-1].max(),f.R[:,-1].max()])*1.1])
            a.set_xticklabels(get_ltx_labels(),rotation=0)
        else:
            a.set_ylim(yl)
            a.set_xticklabels([])
    ax[0].legend(['Input','Fit'])
            
#%% Plot the fit parameters
ylabels=[r'(1-$S^2$)','FWHM',r'$\tau_{max}$ / ps']
clr=['red','green','blue']
for d,title in zip(dist,titles):
    X=list()
    X.append(d['dist'][:,:-1].sum(1)*dz)
    i=np.array([np.argmin(np.abs(z-z0)) for z in d['z']])   #Indicies of the maximum
    ihl=np.array([np.argmin(np.abs(d['dist'][k,i0::-1]-d['A'][k]*0.5)) for k,i0 in enumerate(i)])
    ihr=np.array([np.argmin(np.abs(d['dist'][k,i0:]-d['A'][k]*0.5)) for k,i0 in enumerate(i)])
    X.append(dz*(ihl+ihr))
    X.append(10**d['z']*1e12)
    
    no_data_index=np.concatenate((np.arange(7),[22,30,31]))
    if d is dist[1]:
        for x in X:x[no_data_index]=np.nan
    
    fig=plt.figure()
    ax=[fig.add_subplot(3,1,k) for k in range(1,4)]
    
    for a,c,yl,x in zip(ax,clr,ylabels,X):
        if a==ax[0]:
            a.set_title(title)
        if a==ax[-1]:
            a.semilogy(range(nhg),x[ind['hg']],linestyle='-.',color=c)
            a.semilogy(range(nhg,nhg+nSN2),x[ind['SN2']],linestyle='-',color=c)
            a.semilogy(range(nhg+2,nhg+nSN2),x[ind['SN1']],linestyle=':',color=c)
            a.set_xticks(range(24))
            a.set_xticklabels(get_ltx_labels(),rotation=0)
        else:
            a.plot(range(nhg),x[ind['hg']],linestyle='-.',color=c)
            a.plot(range(nhg,nhg+nSN2),x[ind['SN2']],linestyle='-',color=c)
            a.plot(range(nhg+2,nhg+nSN2),x[ind['SN1']],linestyle=':',color=c)
            a.set_xticks(range(24))
            a.set_xticklabels([])
        a.set_ylabel(yl)

plt.show()    
    
    