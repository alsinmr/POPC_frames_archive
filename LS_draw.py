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



Created on Sat Feb 27 16:00:47 2021

@author: albertsmith
"""

import pyDIFRATE as DR
import numpy as np
import cv2
from DistProd import dist_prod
from lipid_selections import get_indices


#%% NMR sensitivity (use for coloring of Landscapes)
rhoz=DR.io.load_DIFRATE('exper_sens')[:4]
rhoz[0,96:]=0

assert DR.chimeraX.is_chimera_setup(),'Landscape plots require ChimeraX. Please install ChimeraX first, and then run pyDIFRATE.chimeraX.set_chimera_path(path), with path as the location of the ChimeraX executable'

#%% Load data, plot settings, fileouts
def draw_landscape(fr_index=None,mode='NMR'):
    """Draws the dynamic landscape in chimeraX (must have chimeraX properly setup)
    Options are to choose which frame (0-3 for a specific frame, None to draw
    the product of all landscapes). Also, one may draw the NMR-corrected 
    landscape, or the landscape based on MD data only. Set mode='NMR' for the
    NMR corrected landscape (default) or mode='MD' for the MD-only landscape.
    """
    
    dist0=DR.io.load_DIFRATE('LS_NMR_opt' if mode=='NMR' else 'LS_MD_fit')
        
    #%% Extract data, clip
    z0=dist0[0]['z0']
    x0=np.arange(40)        
    dist=dist_prod(z0,np.array([dist_prod(z0,*[d['dist'][k] for d in dist0]) for k in x0])) \
        if fr_index is None else dist0[fr_index]['dist']
    
    if fr_index==1: #These are positions for which there isn't data in frame 1
        dist[[0,1,2,3,4,5,6,22,30,31]]=0
    
    #%% Here we make some corrections of the appearance of the methyl groups
    """
    There's a gap appearing due to the rendering of the methyl group in the 
    perpendicular motion. This is due to a sharp change in the correlation time
    and therefore a failure of the rending. We need to repair this in the rendering
    
    We do so by inserting a few data points in between these  points and 
    extrapolating the distribution in between. 
    """
    
    gap_corr=True
    
    if gap_corr:
        x0=np.insert(x0.astype(float),[21,39],[20.5,38.5])
        i=np.argmax(dist[20,:-10])
        i1=np.argmax(dist[21,:-10])
        shift=int((i-i1)/2)
        A=(dist[20][i]+dist[21][i1])/2/dist[20][i]
        dist=np.insert(dist,21,A*np.concatenate((dist[20][shift:],np.zeros(shift))),axis=0)
        
        i=np.argmax(dist[39,:-10])
        i1=np.argmax(dist[40,:-10])
        shift=int((i-i1)/2)
        A=(dist[39][i]+dist[40][i1])/2/dist[39][i]
        dist=np.insert(dist,40,A*np.concatenate((dist[39][shift:],np.zeros(shift))),axis=0)
    
    "Indices for various parts of the molecule"
    ind=get_indices('eq')
    ind.pop('chains')
    ind=[i for i in ind.values()]
    ind=[ind[0][:3],ind[0][3:6],ind[1],ind[2]]
    ind1=[np.arange(7,22),np.arange(23,30),np.arange(32,40)]
    
    if gap_corr:
        "Adjustments due to rendering correction above"
        ind[2]=np.arange(6,23)
        ind[3]=np.arange(23,42)
        ind1[0]=np.arange(7,23)
        ind1[1]=np.arange(24,31)
        ind1[2]=np.arange(33,42)
    
    "Index for the end of the distribution (cutoff at 1 microsecond)"
    zi=np.argmin(np.abs(z0+6))
    
    "Break distribution into parts"
    dist1=[dist[i,:zi] for i in (ind1 if fr_index==1 else ind)] 
    "(HG,BB,SN1,SN2 or SN1,SN2(before double bond),SN2(after double bond))"
    #%% Calculate correlation-time specific colors
    "Use default matplotlib colors"
    color0=np.array([DR.chimeraX.get_default_colors(k) for k in range(4)])
    "Color corresponding to each correlation time (weighting by the amplitude of detector sensitivities"
    color1=np.array([(c0*rhoz[:,:zi].T).sum(1)/rhoz[:,:zi].sum(0) for c0 in color0.T])
    
    "Color for no amplitude"
    tan=[210,180,140,255]
    
    pw=.25  #Power for color weighting
    color=list()

    maxd=np.max([d.max() for d in dist1]) #Maximum of distribution (rescale coloring with this value)
    color=list()
    for d in dist1:     #Sweep over each part of the molecule
        color.append(list())
        d[d<0]=0    #Zero out any negative contributions (shouldn't be any?)
        for dd in d:    #Sweep over each position
            "Weight tan against color1 based on the scaled detector response"
            color[-1].append(np.dot(np.array([tan]).T,np.array([1-np.power(dd/maxd,pw)]))+np.power(dd/maxd,pw)*color1)
        color[-1]=np.array(color[-1]).swapaxes(0,1)

    
    #%% Linear extrapolation of data for smoothing
    np_sc=5    #Make the data five times longer
    
    "Position offsets, dimension scaling"
    xoff=[-5,-3,2,-16]      #Offsets in position for plot (hg,palmitoyl,oleoyl)
    xoff1=[2,-16,-16]
    zoff=[0,0,5,-4.5]       #Offsets in timescale for plot
    zoff1=[5,-4.5,-4.5]
    zscale=2.5            #Stretching of timescale axis    
    xscale=1.3            #Stretching of position axis
    Ascale=30           #Stretching of vertical axis

    
    de=list()       #Store the scaled amplitudes
    ce=list()       #Store the colors
    xe=list()       #Store the x-axes (position)
    ze=list()       #Store the z-axes (timescale)
    
    
    

        
    if fr_index==1:
        ind0=ind1
        xoff0=xoff1
        zoff0=zoff1
    else:
        ind0=ind
        xoff0=xoff
        zoff0=zoff
    for cc,dd,i,xo,zo in zip(color,dist1,ind0,xoff0,zoff0):
        xnew=np.linspace(x0[i][0]-.5,x0[i][-1]+.5,len(i)*np_sc)
        xe.append((xnew+xo)*xscale)
        ze.append((z0[:zi]+zo)*zscale)
        de.append(DR.tools.linear_ex(x0[i],dd,xnew,mode='last_value')*Ascale)
        ce.append((DR.tools.linear_ex(x0[i],cc,xnew,mode='last_value')).astype('uint8'))
      
    #%% Add in images of the molecule (pngs stored in pngs/POPC-{:02d}.png)
    xrange=np.array([[-6,4],[6.8,24],[5,24]])*xscale-.25
    zoff=np.array([-6.25,-1,-10.5])*zscale+.75
    for k,(xr,zo) in enumerate(zip(xrange,zoff)):
        if not(k==0 and fr_index==1):
            im=cv2.imread('POPC_pngs/POPC-{:02d}.png'.format(k),-1)
            im[:,:,:3]=cv2.cvtColor(im[:,:,:3],cv2.COLOR_BGR2RGB)
            i1=np.all(im<20,axis=-1)
            im[i1]=255
    
            xe.append(np.linspace(xr[0],xr[1],im.shape[1]))
            ze.append(np.linspace(zo,zo+(xr[1]-xr[0])*im.shape[0]/im.shape[1],im.shape[0]))
            de.append(np.zeros(im.shape[:2]).T)
            ce.append(im.T)
            
    #%% Plotting    
    chimera_cmds=['window 1200 1000','set bgColor 15,15,15','lighting soft',\
                  'lighting direction .577,-.577,-.4','lighting shadows true intensity .5',\
                  'lighting ambientIntensity 1.5','material dull','turn x -90',\
                  'turn y -45','turn x 45','view']                        
    
    "Takes a list of surfaces, defined by x-axes, y-axes, amplitudes, and colors"
    "Creates the surface in ChimeraX"        
    DR.cmx_plots.multi_surf(ze,xe,de,colors=ce,chimera_cmds=chimera_cmds)  
        
        
