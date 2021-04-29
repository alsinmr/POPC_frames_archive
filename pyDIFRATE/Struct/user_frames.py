#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:45:12 2020

@author: albertsmith
"""

"""
Use this file to write your own frame definitions. Follow the formating found in
frames.py. A few critical points:
    
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

