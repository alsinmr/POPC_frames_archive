#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:45:12 2020

@author: albertsmith
"""

"""
This module is meant for containing special-purpose frames. Usually, these will
only work on a specific type of system, and may be more complex types of functions
(for example, peptide_plane is still a standard frame, although it only works
for proteins, because it is relatively simple)
    
    1) The first argument must be "molecule", where this refers to the molecule
    object of pyDIFRATE
    2) The output of this function must be another function.
    3) The returned function should not require any input arguments. It should 
    only depend on the current time point in the MD trajectory (therefore, 
    calling this function will return different results as one advances through
    the trajectory).
    4) The output of the sub-function should be one or two vectors (if the
    frame is defined by just a bond direction, for example, then one vector. If
    it is defined by some 3D object, say the peptide plane, then two vectors 
    should be returned)
    5) Each vector returned should be a numpy array, with dimensions 3xN. The
    rows corresponds to directions x,y,z. The vectors do not need to be normalized
    
    6) Be careful of factors like periodic boundary conditions, etc. In case of
    user frames and in the built-in definitions (frames.py) having the same name,
    user frames will be given priority.
    7) The outer function must have at least one required argument aside from
    molecule. By default, calling molecule.new_frame(Type) just returns a list
    of input arguments.
    
    
    Ex.
        def user_frame(molecule,arguments...):
            some_setup
            sel1,sel2,...=molecule_selections (use select_tools for convenience)
            ...
            uni=molecule.mda_object
            
            def sub()
                ...
                v1,v2=some_calculations
                ...
                box=uni.dimensions[:3] (periodic boundary conditions)
                v1=vft.pbc_corr(v1,box)
                v2=vft.pbc_corr(v2,box)
                
                return v1,v2
            return sub
            
"""



import numpy as np
import pyDIFRATE.Struct.vf_tools as vft
import pyDIFRATE.Struct.select_tools as selt


def membrane_grid(molecule,grid_pts,sigma=25,sel0=None,sel='type P',resids=None,segids=None,filter_str=None):
    """
    Calculates motion of the membrane normal, defined by a grid of points spread about
    the simulation. For each grid point, a normal vector is returned. The grid
    is spread uniformly around some initial selection (sel0 is a single atom!)
    in the xy dimensions (currently, if z is not approximately the membrane 
    normal, this function will fail).
    
    The membrane normal is defined by a set of atoms (determined with some 
    combination of the arguments sel, resids, segids, filter_str, with sel_simple)
    
    At each grid point, atoms in the selection will be fit to a plane. However,
    the positions will be weighted depending on how far they are away from that
    grid point in the xy dimensions. Weighting is performed with a normal 
    distribution. sigma, by default, has a width approximately equal to the 
    grid spacing (if x and y box lengths are different, we have to round off the
    spacing)
    
    The number of points is given by grid_pts. These points will be distributed
    automatically in the xy dimensions, to have approximately the same spacing
    in both dimensions. grid_pts will be changed to be the product of the exact
    number of points used (we will always distribute an odd number of points
    in each dimension, so the reference point is in the center of the grid)
    
    if sel0, defining the reference atom, is omitted, then the center of the
    box will be used. Otherwise, the grid will move around with the reference
    atom
    
    membrane_grid(molecule,grid_pts,sigma,sel0,sel,resids,segids,filter_str)
      
    """

    uni=molecule.mda_object
    
    X,Y,Z=uni.dimensions[:3]
    nX,nY=1+2*np.round((np.sqrt(grid_pts)-1)/2*np.array([X/Y,Y/X]))
    dX,dY=X/nX,Y/nY
    
    print('{0:.0f} pts in X, {1:.0f} pts in Y, for {2:.0f} total points'.format(nX,nY,nX*nY))
    print('Spacing is {0:.2f} A in X, {0:.2f} A in Y'.format(dX,dY))
    print('Center of grid is found at index {0:.0f}'.format(nX*(nY-1)/2+(nX-1)/2))
    print('sigma = {0:.2f} A'.format(sigma))
    
    
    if sel0 is not None:
        sel0=selt.sel_simple(molecule,sel0)  #Make sure this is an atom group
        if hasattr(sel0,'n_atoms'):
            if sel0.n_atoms!=1:
                print('Only one atom should be selected as the membrane grid reference point')
                print('Setup failed')
                return
            else:
                sel0=sel0[0]    #Make sure we have an atom, not an atom group
        
        tophalf=sel0.position[2]>Z/2    #Which side of the membrane is this?
    else:
        tophalf=True
        
    "Atoms defining the membrance surface"    
    sel=selt.sel_simple(molecule,sel,resids,segids,filter_str)
    
    "Filter for only atoms on the same side of the membrane"
    sel=sel[sel.positions[:,2]>Z/2] if tophalf else sel[sel.positions[:,2]<Z/2]
    
    def grid():
        "Subfunction, calculates the grid"
        X0,Y0=(X/2,Y/2) if sel0 is None else sel0.position[:2]  #Grid at center, or at position of sel0
        Xout=np.transpose([X0+(np.arange(nX)-(nX-1)/2)*dX]).repeat(nY,axis=1).reshape(int(nX*nY))
        Yout=np.array([Y0+(np.arange(nY)-(nY-1)/2)*dY]).repeat(nY,axis=0).reshape(int(nX*nY))
        return Xout,Yout
    
    def sub():
        "Calculate planes for each element in grid"
        X,Y=grid()
        v=list()
        box=uni.dimensions[:3]
        for x,y in zip(X,Y):  
            v0=vft.pbc_corr(np.transpose(sel.positions-[x,y,0]),box)
            d2=v0[0]**2+v0[1]**2
            i=d2>3*sigma
            weight=np.exp(-d2[i]/(2*sigma**2))
            
            v.append(vft.RMSplane(v0[:,i],np.sqrt(weight)))
        v=np.transpose(v)
        return v/np.sign(v[2])
    
    return sub
    
    
    