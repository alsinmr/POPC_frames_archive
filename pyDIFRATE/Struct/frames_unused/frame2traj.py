#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:24:14 2020

@author: albertsmith
"""

import numpy as np
import os
import vf_tools as vft
import MDAnalysis as mda
from frame_tools import ini_vec_load,apply_index


def remove_frame(molecule,frame,sel,filename,steps=[0,-1,1]):
    """
    Calculates and saves a trajectory for which a given frame is aligned 
    throughout the trajectory. Allows visualization of motion within the frame,
    in the absence of the frame motion itself.
    Arguments are the molecule object, frame, which specifies which frame,
    stored in the molecule object (molecule._vf) should be used (alternatively,
    one may give the function directly), sel, an MDAnalysis atom group that 
    includes all atoms to be stored, filename, which determines where to store
    the trajectory, and steps, which determines the start,stop, and step size 
    over the trajectory (as indices)
    
    Note, the frame should return only a single frame (defined by one or two
    vectors)
    """
    
    uni=molecule.mda_object
    
    if frame is None:
        def f():
            return np.atleast_2d([0,0,1]).T
    else:   
        f=frame if hasattr(frame,'__call__') else molecule._vf[frame]
    
    nv=2 if len(f())==2 else 1

    if steps[1]==-1:steps[1]=uni.trajectory.n_frames
    index=np.arange(steps[0],steps[1],steps[2])

    def pos_fun():
        return (sel.positions-sel.positions.mean(axis=0)).T

    vec=ini_vec_load(uni.trajectory,[f,pos_fun],index=index)

    v1,v2=vec['v'][0] if nv==2 else vec['v'][0],None    
    v1=apply_index(v1,np.zeros(sel.n_atoms))
    if v2 is not None:v2=apply_index(v1,np.zeros(sel.n_atoms))
    v0=vec['v'][1]
    sc=vft.getFrame(v1,v2)
    v0=vft.R(v0,*vft.pass2act(*sc)).T
    
    with mda.Writer(os.path.splitext(filename)[0]+'.pdb',sel.n_atoms) as W:
        sel.positions=v0[0]
        W.write(sel)
    
    with mda.Writer(filename, sel.n_atoms) as W:
        for v in v0:
            sel.positions=v
            W.write(sel)
    
    
def frame_only(molecule,frame_index,sel,filename,steps=[0,-1,1],tstep=0):
    """
    Calculates and saves a trajectory for which a given frame is applied to
    a static selection
    """