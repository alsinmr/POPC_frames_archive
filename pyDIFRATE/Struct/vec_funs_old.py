#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:21:49 2019

@author: albertsmith
"""

import MDAnalysis as mda
import numpy as np
from scipy.linalg import svd

def new_fun(Type,molecule,**kwargs):
    """
    Creates a function to calculate a particular vector(s) from the MD trajectory.
    Mainly responsible for searching the vec_funs file for available functions and
    returning the appropriate function if found (Type determines which function to use)
    
    Required arguments are Type (string specifying the function to be used) and
    a molecule object (contains the MDAnalysis object)
    
    fun=new_fun(Type,molecule,**kwargs)
    
    """
    if Type in globals() and globals()[Type].__code__.co_varnames[0]=='molecule': #Determine if this is a valid vec_fun
        fun0=globals()[Type]
    else:
        raise Exception('Vector function "{0}" was not recognized'.format(Type))
    
    fun=fun0(molecule,**kwargs)
    
    return fun


#%% Vector functions    
def moment_of_inertia(molecule,sel=None,select=None,**kwargs):
    """
    fun=moment_inertia(molecule,sel=None,sel0=None,**kwargs)
    Returns the x, y, and z components of the largest component of the moment 
    of inertia for the given selection.
    
    Note: runs selections function, with arguments sel and select to filter what
    atom groups have the moment of inertial calculated
    """

    sel0=selections(molecule,sel0=sel,select=select)

    def sub():
        vec=list()
        for sel in sel0:
            UT=sel.principal_axes()
            I=sel.moment_of_inertia()
            i=np.diag(UT.dot(I.dot(UT.T))).argmin()
            vec.append([UT[i]])
        return np.concatenate(vec,axis=0).T
    
    return sub


def removeMOI(molecule,sel=None,select=None,**kwargs):
    """
    Factors motion of the moment of inertia out of individual bond motions 
    (effectively removing overall reorienational motion, but not rotational motion)
    By default, factors out motion from all atoms that have the same residue as 
    the first atom in the selection (usually, this should also be the same 
    residue for the second atom)
    
    sel1 and sel2 define the bond direction. These may be previously defined in
    the molecule object, or given here (should be strings that select a set of
    atoms using MDAnalysis style). Nuc may also be used, to select default type of bonds
    
    select is a string that initially filters the selection (for example, 
    select='resname POPC' might pre-select only lipids in a simulation). Argument
    is optional
    
    sel defines a single selection, applied to all bonds, which will define the 
    moment of inertia. sel may be a string, which selects certain atoms in the
    MDanalysis universe, or can be an MDanalysis atom selection (the object itself
    as opposed to the selection string)
    """
    
    mol=molecule

    
    sel1,sel2=selections(molecule,sel1=True,sel2=True,**kwargs)
    
    if sel is None:
        resi0=np.zeros(sel1.n_atoms,dtype=int)
        for k,s in enumerate(sel1):
            resi0[k]=s.resindex
            
        sel=list()
        resi,index=np.unique(resi0,return_index=True) #all residues, and index for each selection
        for r in resi:
            sel.append(mol.mda_object.residues[r].atoms)
    else:
        sel=selections(molecule,sel0=sel,select=select)
        index=np.zeros(sel1.n_atoms)
        resi0=np.zeros(sel1.n_atoms,dtype=int)
        resi=np.zeros(1)

    def sub():
        vec=list()
        for s in sel:
            UT=s.principal_axes()
            I=s.moment_of_inertia()
            i=np.diag(UT.dot(I.dot(UT.T))).argmin()
            vec.append(UT[i])
            
        vec0=sel1.positions-sel2.positions
        
        box=mol.mda_object.dimensions[0:3]
        
        vec0=pbc_corr(vec0,box)
        vec_out=np.zeros(vec0.shape)
        
        for resi1,vec1 in zip(resi,vec):
            i=resi1==resi0
            vec_out[i]=remove(vec0[i],vec1).T
        
        return vec_out                
    return sub

def removePP(molecule,resi=None):
    """
    Separates peptide plane motion from NH motion, leaving only the librational
    motion of the NH bond. molecule can already contain sel1 and sel2, which 
    should be N and H atoms of a molecule. Alternatively, one may simply specify
    the desired residues. If mol.sel1 and mol.sel2 are both None and no resi
    is supplied, then all NH bonds are analyzed
    """
    
    mol=molecule
    uni=mol.mda_object
    "Get the selections"
    if resi is None and mol.sel1 is None and mol.sel2 is None:
        resi=uni.atoms.resids[1:-1].tolist()
    elif resi is None:
        resi=list()
        for s in molecule.sel1:
            resi.append(s.resnum)
    
    sel0=uni.atoms.select_atoms('protein')
    i=list()
    for k,r in enumerate(sel0.residues):
        if r.resid in resi and ('HN' in r.atoms.names or 'H' in r.atoms.names):
            i.append(k)
    i=np.array(i)
    
    resi=list()
    for i0 in i:
        resi.append(sel0.residues[i0].atoms[0].resid)
    mol.label=np.array(resi)

    sel0=sel0.residues[i].atoms

    sel1=sel0.select_atoms('name H or name HN')
    sel2=sel0.select_atoms('name N')
    mol.sel1=sel1
    mol.sel2=sel2
    
    sel0=uni.atoms.select_atoms('protein').residues[i-1].atoms
    
    pln1=sel0.select_atoms('name O')
    pln2=sel0.select_atoms('name C')   

             
    
    def sub():
        box=uni.dimensions[0:3]
        v0=sel1.positions-sel2.positions
        v1=pln1.positions-pln2.positions
        v2=sel2.positions-pln2.positions
        v0=pbc_corr(v0,box)
        v1=pbc_corr(v1,box)
        v2=pbc_corr(v2,box)
        v=removePlane(v0,v1,v2)
        return v
    
    return sub

def RotOnly(molecule,sel1=True,sel2=True,**kwargs):
    """
    Selects librational and rotational motion for a bond, defined by sel1 and 
    sel2. In case of librational motion, this is done by working in a frame for 
    which bonds connecting the heteroatom of the bond to other heteroatoms are 
    aligned along z (the first bond), and fixed in the xz plane (the second bond).
    
    In the case of terminal carbons (methyl groups, or other groups bonded only
    to one heteroatom), the bond to the heteroatom will be fixed along z
    
    sel1 and sel2 may be defined directly (as strings or MDAnalysis atom groups),
    or if not defined, they will be taken from the molecule object
    
    f=LibRot(molecule,sel1=True,sel2=True,**kwargs)
    
    Note that we assume that only sel1 or sel2 contains heteroatoms (for example,
    if HN bonds are selected, presumably all the Ns are either in sel1 or sel2)
    """
    
    uni=molecule.mda_object
    uni.trajectory.rewind()
    
    sel1,sel2=selections(molecule,sel1=sel1,sel2=sel2,**kwargs)
    
    if sel2[0].mass>=3: #Require that sel1 contains the heteroatoms
        sel1,sel2=sel2,sel1
    
    sel0=sel1.universe.atoms #All atoms
    sel0=sel0[sel0.masses>3]    #Heteroatoms
    
    i=list()
    j=list()
    methyl=list()
    for s in sel1:
        if sel0.n_segments>1:
            s0=sel0[sel0.segids==s.segid]   #Just the atoms in the same segment
        else:
            s0=sel0[sel0.resids==s.resid]   #Just the atoms in the same residue
        s0=sel0
        d=np.square(s0.positions-s.position).sum(axis=1)
        i0=np.argsort(d)
        i.append(s0[i0[1]].id) #Closest heteroatom (excluding self)
        j.append(s0[i0[2]].id) #Second closest heteroatom (excluding self)
        if d[i0[2]]>(1.75**2): #Flag for methyl group (no bonded heteroatom, only remove rotation)
            methyl.append(True)
        else:
            methyl.append(False)
    
    methyl=np.array(methyl,dtype=bool)
    nm=np.logical_not(methyl)
    ref1=uni.atoms[np.array(i)-1]   #Indexing shift
    ref2=uni.atoms[np.array(j)-1]
    
    v=np.zeros([3,methyl.shape[0]])
    
    v0=sel1[nm].positions-sel2[nm].positions
    v1=ref1[nm].positions-ref2[nm].positions
    
    beta=np.arccos((v0*v1).sum(axis=1)/np.sqrt((v0**2).sum(axis=1)*(v1**2).sum(axis=1)))
    beta[:]=np.pi/2
    
    v0=sel1[methyl].positions-sel2[methyl].positions
    v1=sel1[methyl].positions-ref1[methyl].positions
    
    betam=np.arccos((v0*v1).sum(axis=1)/np.sqrt((v0**2).sum(axis=1)*(v1**2).sum(axis=1)))
    betam[:]=109.5*np.pi/180
    
    def sub():
        box=uni.dimensions[0:3]
        v0=sel1[nm].positions-sel2[nm].positions
        v1=ref1[nm].positions-ref2[nm].positions
        v0=pbc_corr(v0,box)
        v1=pbc_corr(v1,box)
        "Should beta be fixed at pi/2?????"
        v00=remove(v0,v1,beta) 
        
        v0=sel1[methyl].positions-sel2[methyl].positions
        v1=sel1[methyl].positions-ref1[methyl].positions
        v0=pbc_corr(v0,box)
        v1=pbc_corr(v1,box)
        v01=remove(v0,v1,betam)
        
        
        v[:,nm]=v00
        v[:,methyl]=v01
        
        return v
    
    return sub        

def NoRot(molecule,sel1=True,sel2=True,**kwargs)    :
    
    uni=molecule.mda_object
    
    sel1,sel2=selections(molecule,sel1=sel1,sel2=sel2,**kwargs)
    
    if sel2[0].mass>=3: #Require that sel1 contains the heteroatoms
        sel1,sel2=sel2,sel1
    
    sel0=sel1.universe.atoms #All atoms
    sel0=sel0[sel0.masses>3]    #Heteroatoms
    
    i=list()
    j=list()
    methyl=list()
    for s in sel1:
        if sel0.n_segments>1:
            s0=sel0[sel0.segids==s.segid]   #Just the atoms in the same segment
        else:
            s0=sel0[sel0.resids==s.resid]   #Just the atoms in the same residue
        s0=sel0
        v0=s0.positions-s.position
        v0=pbc_corr(v0,uni.dimensions[0:3])
        d=np.square(v0).sum(axis=1)
        i0=np.argsort(d)
        i.append(s0[i0[1]].id) #Closest heteroatom
        j.append(s0[i0[2]].id) #Second closest heteroatom
        if d[i0[2]]>(1.75**2): #Flag for methyl group (no bonded heteroatom, only remove rotation)
            methyl.append(True)
        else:
            methyl.append(False)
            
    methyl=np.array(methyl,dtype=bool)
    nm=np.logical_not(methyl)
    ref1=uni.atoms[np.array(i)-1]   #Indexing shift
    ref2=uni.atoms[np.array(j)-1]
    
    v0=sel1[nm].positions-sel2[nm].positions
    v1=ref1[nm].positions-ref2[nm].positions
    
    beta=np.arccos((v0*v1).sum(axis=1)/np.sqrt((v0**2).sum(axis=1)*(v1**2).sum(axis=1)))
    beta[:]=np.pi/2
    
    v=np.zeros([3,methyl.shape[0]])

    def sub():
        box=uni.dimensions[0:3]
        v0=sel1[nm].positions-sel2[nm].positions
        v1=ref1[nm].positions-ref2[nm].positions
        v0=pbc_corr(v0,box)
        v1=pbc_corr(v1,box)
        v00=remove_rot(v0,v1,beta)
        
        v01=ref1[methyl].positions-sel1[methyl].positions
        v01=pbc_corr(v01,box)
        
        v[:,nm]=v00
        v[:,methyl]=v01.T
        
        return v
    
    return sub

def PP(molecule,resi=None):
    """
    Calculates peptide plane motion
    """
    mol=molecule
    uni=mol.mda_object
    "Get the selections"
    if resi is None and mol.sel1 is None and mol.sel2 is None:
        resi=uni.atoms.resids[1:-1].tolist()
    elif resi is None:
        resi=list()
        for s in mol.sel1:
            resi.append(s.resnum)
    
    sel0=uni.atoms.select_atoms('protein')
    i=list()
    for k,r in enumerate(sel0.residues):
        if r.resid in resi and ('HN' in r.atoms.names or 'H' in r.atoms.names):
            i.append(k)
    i=np.array(i)
    
    resi=list()
    for i0 in i:
        resi.append(sel0.residues[i0].atoms[0].resid)
    mol.label=np.array(resi)
    
    sel0=sel0.residues[i].atoms

    sel1=sel0.select_atoms('name H or name HN')
    sel2=sel0.select_atoms('name N')
    mol.sel1=sel1
    mol.sel2=sel2
    
    sel0=uni.atoms.select_atoms('protein').residues[i-1].atoms
    
    pln1=sel0.select_atoms('name CA')
    pln2=sel0.select_atoms('name C')
    
    v0=sel1.positions-sel2.positions
    v1=pln1.positions-pln2.positions
    v2=pln2.positions-sel2.positions
    
    box=uni.dimensions[0:3]
    v0=pbc_corr(v0,box)
    v1=pbc_corr(v1,box)
    v2=pbc_corr(v2,box)
    
    d=list()
    for a,b,c in zip(v0,v1,v2):
        d.append(np.linalg.lstsq(np.array([b,c]).T,a,rcond=None)[0])
    
    wt=np.array(d).mean(axis=0)
    print(wt)
    def sub():
        box=uni.dimensions[0:3]
        v1=pln1.positions-pln2.positions
        v2=pln2.positions-sel2.positions
        v1=pbc_corr(v1,box)
        v2=pbc_corr(v2,box)
        
        v=wt[0]*v1+wt[1]*v2
        
        return v.T
    return sub
        
#def PhiPsi(molecule,resi=None):
#    """
#    Separates Phi-Psi dynamics from overall dynamics
#    """
#    
#    mol=molecule
#    uni=mol.mda_object
#    sel0=uni.atoms.select_atoms('protein')
#    
#    "Get the selections"
#    if resi is None and mol.sel1 is None and mol.sel2 is None:
#        resi=np.unique(sel0.resids)[0:-1].tolist()
#    elif resi is None:
#        resi=list()
#        for s in mol.sel1:
#            resi.append(s.resnum)
#        resi=np.unique(resi).tolist()
#    
#    i=list()
#    for k,r in enumerate(sel0.residues):
#        if r.resid in resi and ('HN' in r.atoms.names or 'H' in r.atoms.names):
#            i.append(k)
#    i=np.array(i)
#
#    
#    sel_phi=list()
#    sel_phi.append(sel0.residues[i-1].atoms.select_atoms('name C'))
#    sel_phi.append(sel0.residues[i].atoms.select_atoms('name N'))
#    sel_phi.append(sel0.residues[i].atoms.select_atoms('name CA'))
#    sel_phi.append(sel0.residues[i].atoms.select_atoms('name C'))
#    
#    sel_psi=list()
#    sel_psi.append(sel0.residues[i].atoms.select_atoms('name N'))
#    sel_psi.append(sel0.residues[i].atoms.select_atoms('name CA'))
#    sel_psi.append(sel0.residues[i].atoms.select_atoms('name C'))
#    sel_psi.append(sel0.residues[i+1].atoms.select_atoms('name N'))
#    
#
#    
#    def sub():
#        box=uni.dimensions[0:3]
#        phi=dihedral(sel_phi[0],sel_phi[1],sel_phi[2],sel_phi[2],box)
#        psi=dihedral(sel_psi[0],sel_psi[1],sel_psi[2],sel_psi[2],box)
#        
#        v=[phi,psi]
#        return v
#    
#    return sub    

def PhiPsi(molecule,resi=None):
    """
    Separates Phi-Psi dynamics from overall dynamics
    """
    
    mol=molecule
    uni=mol.mda_object
    sel0=uni.atoms.select_atoms('protein')
    
    "Get the selections"
    if resi is None and mol.sel1 is None and mol.sel2 is None:
        resi=np.unique(sel0.resids)[0:-1].tolist()
    elif resi is None:
        resi=list()
        for s in mol.sel1:
            resi.append(s.resnum)
        resi=np.unique(resi).tolist()
    
    i=list()
    for k,r in enumerate(sel0.residues):
        if r.resid in resi and ('HN' in r.atoms.names or 'H' in r.atoms.names):
            i.append(k)
    i=np.array(i)

    resi=list()
    for i0 in i:
        resi.append(sel0.residues[i0].atoms[0].resid)
    mol.label=np.array(resi)

    sel0=sel0.residues[i].atoms

    sel1=sel0.select_atoms('name H or name HN')
    sel2=sel0.select_atoms('name N')
    mol.sel1=sel1
    mol.sel2=sel2
    
    pln1=sel0.select_atoms('name O')
    pln2=sel0.select_atoms('name C')            
    pln3=sel0.select_atoms('name CA')
    
    def sub():
        box=uni.dimensions[0:3]
        v0=sel1.positions-sel2.positions
        v1=pln2.positions-pln3.positions
        v2=pln1.positions-pln2.positions
        v0=pbc_corr(v0,box)
        v1=pbc_corr(v1,box)
        v2=pbc_corr(v2,box)
        v=removePlane(v0,v1,v2)
        return v
    
    return sub

def rot_axis(molecule,axis=[0,0,1],**kwargs):
    """
    Calculates the rotation of a vector (defined by a pair of atoms) around a 
    fixed axis (for example, the z-axis of the MD axis system). Must provide the
    molecule object and the axis. The selection of atoms may be defined here 
    (using the same format as molecule.select_atoms), or if a selection is already
    made with molecule.select_atoms, one may simply give the molecule and axis here
    """    
    
#    if 'sel1' in kwargs or 'sel2' in kwargs or 'Nuc' in kwargs:
#        molecule.select_atoms(**kwargs) #Run selection if appropriate arguments are provided
#    
#    axis=np.atleast_1d(axis)
#    axis=axis/np.sum(axis**2)
#    
#    sel1=molecule.sel1
#    sel2=molecule.sel2
#    if molecule.sel1in is None:
#        sel1in=np.arange(sel1.n_atoms)
#    else:
#        sel1in=molecule.sel1in
#    if molecule.sel2in is None:
#        sel2in=np.arange(sel2.n_atoms)
#    else:
#        sel2in=np.molecule.sel2in
#   
    if 'sel1' not in kwargs:
        kwargs.update({'sel1':True})
    if 'sel2' not in kwargs:
        kwargs.update({'sel2':True})
    a,b=selections(molecule,**kwargs)
    sel1,sel1in=a
    sel2,sel2in=b
    
    def sub():
        vec0=sel1[sel1in].positions-sel2[sel2in].positions
        vec=project(vec0,axis,Type='plane')
        return vec
    
    return sub

def PPaligned(molecule,resi=None,sel=None):
    """
    Calculates peptide plane motion in an aligned reference frame, defined by
    a selection string
    
    
    """
    
    mol=molecule
    traj=mol.mda_object.trajectory
    sel0=mol.mda_object.select_atoms('protein')
    
    if resi is None:
        resi=np.unique(sel0.resindices)
        
    if sel is None:
        sel=sel0.select_atoms('name CA')
    elif isinstance(sel,str):
        sel=sel0.select_atoms(sel)
       
    traj.rewind()
    pos=sel.positions
    vref=pos-pos.mean(axis=0)
    
    vPP=PP(molecule,resi)
    
    def sub():
        pos=sel.positions
        v=pos-pos.mean(axis=0)
        
        R=alignR(v,vref)
        
        return np.dot(R,vPP())
    
    return sub
        

def AlignedFrame(molecule,sel1=True,sel2=True,sel=None,select=None,**kwargs):
    """
    Aligns a selection of atoms to create a new reference frame (remove overall
    motion)
    
    sel1 and sel2 define the bond direction. These may be previously defined in
    the molecule object, or given here (should be strings that select a set of
    atoms using MDAnalysis style). Nuc may also be used, to select default type of bonds
    
    select is a string that initially filters the selection (for example, 
    select='resname POPC' might pre-select only lipids in a simulation). Argument
    is optional
    
    sel defines a single selection, applied to all bonds, which will define the 
    moment of inertia. sel may be a string, which selects certain atoms in the
    MDanalysis universe, or can be an MDanalysis atom selection (the object itself
    as opposed to the selection string)
    """
    mol=molecule
    traj=mol.mda_object.trajectory

    sel1,sel2=selections(molecule,sel1,sel2,**kwargs)
    
    if sel is None:
        resi0=np.zeros(sel1.n_atoms,dtype=int)
        for k,s in enumerate(sel1):
            resi0[k]=s.resindex
            
        sel=list()
        resi,index=np.unique(resi0,return_index=True) #all residues, and index for each selection
        for r in resi:
            sel.append(mol.mda_object.residues[r].atoms)
            if 'CA' in sel[-1].names:
                sel[-1]=sel[-1].select_atoms('name CA')
    else:
        sel=[selections(molecule,sel0=sel,select=select)]
#        sel,_=a
        index=np.zeros(sel1.n_atoms)
        resi0=np.zeros(sel1.n_atoms,dtype=int)
        resi=np.zeros(1)

    
    #%% Get a set of reference vectors for each selection in sel
    vref=list()
    traj.rewind()
    for s in sel:
        box=traj.dimensions[0:3]
        pos=s.positions
        vref.append(pos-pos.mean(axis=0))

    
    def sub():
        R=list()
        for s,vr in zip(sel,vref):
            pos=s.positions
            box=traj.dimensions[0:3]
            v=pos-pos.mean(axis=0)
            R.append(alignR(v,vr))
            
        vec0=sel1.positions-sel2.positions
        
        box=traj.dimensions[0:3]
        
        vec0=pbc_corr(vec0,box)
        vec_out=np.zeros(vec0.shape)
        
        for resi1,R1 in zip(resi,R):
            i=resi1==resi0
            vec_out[i]=np.dot(vec0[i],R1.T)
        
        return vec_out.T                
    return sub

#%% Supporting functions
def pbc_corr(v0,box):
    v=np.atleast_2d(v0)
    box=np.repeat([box],v.shape[0],axis=0)
            
    i=v>box/2
    v[i]=v[i]-box[i]
    
    i=v<-box/2
    v[i]=v[i]+box[i]
    
    if np.ndim(v0)==1:
        v=v[0]
        
    return v


def dihedral(p1,p2,p3,p4,box=None):
    """
    Calculates a dihedral angle from 4 atom positions
    """
    #Get vectors
    v1=p2-p1
    v2=p3-p2
    v3=p4-p3
    #Correct for PBC
    if box is not None:
        v1=pbc_corr(v1,box)
        v2=pbc_corr(v2,box)
        v3=pbc_corr(v3,box)
    
    v2=v2/np.sum(v2**2,axis=1)  #We need v2 normalized
    
    n1=np.cross(v1,v2)
    n2=np.cross(v2,v3)
    
    m1=np.cross(n1,v2)
    
    x=np.dot(n1,n2)
    y=np.dot(m1,n2)
    
    return np.arctan2(y,x)

def alignR(v0,vref):
    """
    Returns the optimal rotation matrix to rotate a set of vectors v0 to a set 
    of reference vectors, vref
    
    R=alignR(v0,vref)
    
    Uses the Kabsch algorithm. Assumes *vectors*, with origins at zero, not 
    points, so that no translation will be performed
    """
    
    H=np.dot(v0.T,vref)
    
    U,S,Vt=svd(H)
    V=Vt.T
    Ut=U.T
    
    d=np.linalg.det(np.dot(V,Ut))
    m=np.eye(3)
    m[2,2]=d
    
    R=np.dot(V,np.dot(m,Ut))
    
    return R

def removePlane(vec0,vec1,vec2):
    """
    Removes plane motion from a given vector selection, for example one can remove
    peptide plane motion from the motion of an NH bond, or chain motion in a lipid
    from a given CH bond.
    
    Typically, this will be used to separate fast, librational motions from the
    more interesting slow motions
    
    vec=removePlane(vec,vec1,vec2)
    
    vec0 will be transformed to vec, where vec1 and vec2 define the plane. 
    
    Our procedure will be to find euler angles to rotate the first reference 
    vector to the z-axis and the second reference vector to be contained in the
    xz plane.
    """
    
    
    """
    For completeness, all vec calcs are below, however, those that are not
    actually needed are commented out.
    """
    
    if vec0.shape[0]!=3:
        vec0=vec0.T
    if vec1.shape[0]!=3:
        vec1=vec1.T
    if vec2.shape[0]!=3:
        vec2=vec2.T
    
    gamma=np.arctan2(vec1[1],vec1[0])   #Rotates vec1 so that there is no y component    
#    x1=vec1[0]*np.cos(gamma)+vec1[1]*np.sin(gamma)
#    y1=-vec1[0]*np.sin(gamma)+vec1[1]*np.cos(gamma)    
    x2=vec2[0]*np.cos(gamma)+vec2[1]*np.sin(gamma)
    y2=-vec2[0]*np.sin(gamma)+vec2[1]*np.cos(gamma)    
    x0=vec0[0]*np.cos(gamma)+vec0[1]*np.sin(gamma)
    y0=-vec0[0]*np.sin(gamma)+vec0[1]*np.cos(gamma)
     
    l1=np.sqrt(np.sum(vec1**2,axis=0))
    beta=np.arccos(vec1[2]/l1)      #Rotates vec1 so that there is only a z-component  
#    x10=x1*np.cos(beta)-vec1[2]*np.sin(beta)
#    z1=x1*np.sin(beta)+vec1[2]*np.cos(beta)
#    x1=x10
    x20=x2*np.cos(beta)-vec2[2]*np.sin(beta)
#    z2=x2*np.sin(beta)+vec2[2]*np.cos(beta)
    x2=x20
    x00=x0*np.cos(beta)-vec0[2]*np.sin(beta)
    z0=x0*np.sin(beta)+vec0[2]*np.cos(beta)
    x0=x00
    
    alpha=np.arctan2(y2,x2) #Rotates vec2 so that there is no y component
#    x10=x1*np.cos(alpha)+y1*np.sin(alpha)
#    y1=-x1*np.sin(alpha)+y1*np.cos(alpha)
#    x1=x10 
#    x20=x2*np.cos(alpha)+y2*np.sin(alpha)
#    y2=-x2*np.sin(alpha)+y2*np.cos(alpha)
#    x2=x20
    x00=x0*np.cos(alpha)+y0*np.sin(alpha)
    y0=-x0*np.sin(alpha)+y0*np.cos(alpha)
    x0=x00
    
    vec=np.array([x0,y0,z0])
    
    return vec
    
#def remove(vec0,vec1):
#    """
#    Removes motion of one vector from the motion of the other. That is, we find
#    a transformation to rotate vec1 to the z-axis, and apply the same transformation
#    to vec0. vec0 may be an array of vectors, and vec1 may be a single vector or
#    an array of vectors the same size as vec0
#
#    vec=remove(vec0,vec1)
#    """    
#    vec0=np.array(vec0)
#    vec1=np.array(vec1)
#    
#    if vec0.shape[0]!=3:
#        vec0=vec0.T
#    if vec1.shape[0]!=3:
#        vec1=vec1.T
#    
#    vec1=vec1/np.sqrt(np.sum(vec1**2,axis=0))
#    
#    theta=np.arctan2(vec1[0],vec1[1])
#    phi=np.arccos(vec1[2])
#    
#    vec=np.zeros(vec0.shape)
#    
#    x=vec0[0]*np.cos(theta)-vec0[1]*np.sin(theta)
#    y0=vec0[0]*np.sin(theta)+vec0[1]*np.cos(theta)
#    z0=vec0[2]
#    z=z0*np.cos(phi)+y0*np.sin(phi)
#    y=-z0*np.sin(phi)+y0*np.cos(phi)
#    
#
#    return np.array([x,y,z])

def remove(vec0,vec1,beta=None):
    """
    Removes motion of one vector from the motion of the other. That is, we find
    a transformation to rotate vec1 to the z-axis, and apply the same transformation
    to vec0. vec0 may be an array of vectors, and vec1 may be a single vector or
    an array of vectors the same size as vec0

    vec=remove(vec0,vec1)
    """    
    vec0=np.array(vec0)
    vec1=np.array(vec1)
    
    if vec0.shape[0]!=3:
        vec0=vec0.T
    if vec1.shape[0]!=3:
        vec1=vec1.T
    
    vec1=vec1/np.sqrt(np.sum(vec1**2,axis=0))
    
    theta=np.arctan2(vec1[1],vec1[0])
    phi=np.arccos(vec1[2])
    

    x,y,z=vec0
    "Apply theta"    
    x,y=x*np.cos(theta)+y*np.sin(theta),-x*np.sin(theta)+y*np.cos(theta)
    "Apply phi"
    x,z=x*np.cos(phi)-z*np.sin(phi),x*np.sin(phi)+z*np.cos(phi)
    if beta is not None:    #This will remove librational motion
        l=np.sqrt(x**2+y**2+z**2)
        z=l*np.cos(beta) #We fix the angle between vec0 and vec1! (vec1 is along z at this step)
        x,y=[x,y]/np.sqrt(x**2+y**2)*l*np.sin(beta)
    "Reverse theta rotation"
    x,y=x*np.cos(-theta)+y*np.sin(-theta),-x*np.sin(-theta)+y*np.cos(-theta)
    

    return np.array([x,y,z])

#def remove_rot(vec0,vec1):
#    """
#    Removes rotation of vec0 around vec1, leaving other motions induced on vec0
#    by the re-orientation of vec1. The procedure is to find alpha, beta, and 
#    gamma to tranform vec1 to the z-axis and vec0 into the xz-plane. These are
#    applied to vec0, but then the reverse transformation is applied, while 
#    omiting alpha, which represents the rotation around vec1
#    """
#    
#    if vec0.shape[0]!=3:
#        vec0=vec0.T
#    if vec1.shape[0]!=3:
#        vec1=vec1.T
#    
#    vec1=vec1/np.sqrt(np.sum(vec1**2,axis=0))
#    
#    
#    gamma=np.arctan2(vec1[1],vec1[0])   #Rotates vec1 so that there is no y component    
##    x1=vec1[0]*np.cos(gamma)+vec1[1]*np.sin(gamma)
##    y1=-vec1[0]*np.sin(gamma)+vec1[1]*np.cos(gamma)    
#    x0=vec0[0]*np.cos(gamma)+vec0[1]*np.sin(gamma)
#    y0=-vec0[0]*np.sin(gamma)+vec0[1]*np.cos(gamma)
#     
#    beta=np.arccos(vec1[2])      #Rotates vec1 so that there is only a z-component  
##    x10=x1*np.cos(beta)-vec1[2]*np.sin(beta)
##    z1=x1*np.sin(beta)+vec1[2]*np.cos(beta)
##    x1=x10
#    x00=x0*np.cos(beta)-vec0[2]*np.sin(beta)
#    z0=x0*np.sin(beta)+vec0[2]*np.cos(beta)
#    x0=x00
#    
#    alpha=np.arctan2(y0,x0) #Rotates vec2 so that there is no y component
##    x10=x1*np.cos(alpha)+y1*np.sin(alpha)
##    y1=-x1*np.sin(alpha)+y1*np.cos(alpha)
##    x1=x10 
#    x00=x0*np.cos(alpha)+y0*np.sin(alpha)
#    y0=-x0*np.sin(alpha)+y0*np.cos(alpha)
#    x0=x00
#    
#    "Now we rotate back, omitting alpha"
#    
#    x00=x0*np.cos(-beta)-z0*np.sin(-beta)
#    z0=x0*np.sin(-beta)+z0*np.cos(-beta)
#    x0=x00
#    
#    x00=x0*np.cos(-gamma)+y0*np.sin(-gamma)
#    y0=-x0*np.sin(-gamma)+y0*np.cos(-gamma)
#    x00=x0
#    
#    return np.array([x0,y0,z0])

def remove_rot(vec0,vec1,phi=None):
    """
    Removes rotation of vec0 around vec1, leaving other motions induced on vec0
    by the re-orientation of vec1. The procedure is to find alpha, beta, and 
    gamma to tranform vec1 to the z-axis and vec0 into the xz-plane. These are
    applied to vec0, but then the reverse transformation is applied, while 
    omiting alpha, which represents the rotation around vec1
    """
    
    if vec0.shape[0]!=3:
        vec0=vec0.T
    if vec1.shape[0]!=3:
        vec1=vec1.T
    
    vec1=vec1/np.sqrt(np.sum(vec1**2,axis=0))
    
    "Calculate and apply gamma"
    x0,y0,z0=vec0
    x1,y1,z1=vec1
    gamma=np.arctan2(y1,x1)   #Rotates vec1 so that there is no y component    
#    x1,y1=x1*np.cos(gamma)+y1*np.sin(gamma),-x1*np.sin(gamma)+y1*np.cos(gamma)     
    x0,y0=x0*np.cos(gamma)+y0*np.sin(gamma),-x0*np.sin(gamma)+y0*np.cos(gamma)
    
    "Calculate and apply beta"
    beta=np.arccos(z1)      #Rotates vec1 so that there is only a z-component  
#    x1,z1=x1*np.cos(beta)-z1*np.sin(beta),x1*np.sin(beta)+z1*np.cos(beta)
    x0,z0=x0*np.cos(beta)-z0*np.sin(beta),x0*np.sin(beta)+z0*np.cos(beta)
    
    "-gamma"
    x0,y0=x0*np.cos(-gamma)+y0*np.sin(-gamma),-x0*np.sin(-gamma)+y0*np.cos(-gamma)
    
    "Calculate and apply alpha"
    alpha=np.arctan2(y0,x0) #Rotates vec0 so that there is no y component
#    x1,y1=x1*np.cos(alpha)+y1*np.sin(alpha),-x1*np.sin(alpha)+y1*np.cos(alpha)
    x0,y0=x0*np.cos(alpha)+y0*np.sin(alpha),-x0*np.sin(alpha)+y0*np.cos(alpha)
    
    if phi is not None:
        l=np.sqrt(x0**2+y0**2+z0**2)
        x0,z0=[np.sin(phi),np.cos(phi)]*l
        
    
    "Now we rotate back, omitting alpha"
    "gamma"
    x0,y0=x0*np.cos(gamma)+y0*np.sin(gamma),-x0*np.sin(gamma)+y0*np.cos(gamma)
    "-beta"
    x0,z0=x0*np.cos(-beta)-z0*np.sin(-beta),x0*np.sin(-beta)+z0*np.cos(-beta)
    "-gamma"
    x0,y0=x0*np.cos(-gamma)+y0*np.sin(-gamma),-x0*np.sin(-gamma)+y0*np.cos(-gamma)
    
    
    return np.array([x0,y0,z0])

def selections(molecule,sel0=None,sel1=None,sel2=None,select=None,Nuc=None,sel1in=None,sel2in=None):
    """
    General function for returning a set of selections (does not edit the molecule
    object)
    One may return up to three selections, specified by sel0, sel1, and sel2.
    Each argument may be given as a string (valid for atom selection in MDAnalysis),
    as an MDAnalysis selection, or as a list of MDAnalysis selections. sel1 and sel2
    may alternatively be set to True, which will then return the selection already
    stored in molecule.sel1 or molecule.sel2 (depending on if sel1 or sel2 is set
    to True). All selections will be filtered with the selection string (valid
    for MDAnalysis) given in select (if not set to None, which is default)
    
    (s0,s1,s2)=selections(molecule,sel0=None,sel1=None,sel2=None,select=None)
    
    Note, the size of the tuple returned by selections depends on whether sel0,
    sel1, and/or sel2 are set to None (for example, if only sel0 is not None, then
    only one selection is returned)
    
    sel0 is returned as a list of selections (in some cases, multiple selections are needed)
    sel1 and sel2 are returned as single selections, but are a tuple including the
    selection and an index. The index is usually just np.arange(sel1.n_atoms), a
    list numbering from 0 up to the number of atoms in the selection, but if an
    index is provided in mol.sel1in, this may be passed, or one may provide sel1in and
    sel2in directly to selection (simply passed to output)
    
    Finally, note that if sel0, sel1, and sel2 are all None, then a set of selections
    will be returned, where all atoms in each segment or each molecule (if only one
    segment in simulation, will then try residues)
    """
    
    mol=molecule
    
    "Get sel1 and/or sel2 out of molecule in case"
    if isinstance(sel1,bool) and sel1:
        sel1=mol.sel1
        if mol.sel1in is not None:
            sel1=sel1[mol.sel1in]
    if isinstance(sel2,bool) and sel2:
        sel2=mol.sel2
        if mol.sel2in is not None:
            sel2=sel2[mol.sel2in]
     
    if Nuc is not None:
        sel1,sel2=nuc_defaults(Nuc)
    
    if sel0 is not None:
        sel0=sel_helper(molecule,sel0,select)
    if sel1 is not None:
        sel1=sel_helper(molecule,sel1,select)  
    if sel2 is not None:
        sel2=sel_helper(molecule,sel2,select)
            
    if sel0 is None and sel1 is None and sel2 is None:
        sel0=sel_helper(molecule,None,select)
    
    "If an index provided, then apply that index now"
    if sel1in is not None and sel1 is not None:
        sel1=sel1[sel1in]
    if sel2in is not None and sel2 is not None:
        sel2=sel2[sel2in]
    
    if sel0 is None and sel1 is None:
        return sel2
    elif sel0 is None and sel2 is None:
        return sel1
    elif sel1 is None and sel2 is None:
        return sel0
    elif sel0 is None:
        return (sel1,sel2)
    elif sel1 is None:
        return (sel0,sel2)
    elif sel2 is None:
        return (sel0,sel1)
            
def sel_helper(molecule,sel,select):
    uni=molecule.mda_object
    if isinstance(sel,list):
        "If list provided, just run over all values, and drop into a new list"
        sel_out=list()
        for sel1 in sel:
            sel_out.append(sel_helper(molecule,sel1,select))
    elif sel is None:
        "If no selection provided, break up by segments or residues"
        if select is None:
            sel0=uni.atoms
        else:
            sel0=uni.select_atoms(select)
        sel_out=list()
        if sel0.n_segments>1:
            for a in sel0.segments:
                sel_out.append(a.atoms)
        elif sel0.n_residues>1:
            for a in sel0.residues:
                sel_out.append(a.atoms)
        else:
            sel_out.append(sel0)
    elif isinstance(sel,str):
        "Make selection with a string"
        if select is None:
            sel0=uni.atoms
        else:
            sel0=uni.select_atoms(select)
            
        sel_out=sel0.select_atoms(sel)
    elif isinstance(sel.atoms,mda.AtomGroup):
        if select is None:
            sel_out=sel
        else:
            sel_out=sel.select_atoms(select)
            
    return sel_out

def nuc_defaults(Nuc):
    if Nuc.lower()=='15n' or Nuc.lower()=='n' or Nuc.lower()=='n15':
            sel1='(name H or name HN) and around 1.1 name N'
            sel2='name N and around 1.1 (name H or name HN)'
    elif Nuc.lower()=='co':
        sel1='name C and around 1.4 name O'
        sel2='name O and around 1.4 name C'
    elif Nuc.lower()=='ca':
        sel1='name CA and around 1.4 (name HA or name HA2)'
        sel2='(name HA or name HA2) and around 1.4 name CA'
        print('Warning: selecting HA2 for glycines. Use manual selection to get HA1 or both bonds')
        
    return (sel1,sel2)


#%% Functions to be removed
"I suspect these are useless, but don't want to delete them yet"
"Projections just yield the vector being projected onto, so there is little point"

def project(vec0,vec1,Type='plane'):
    """
    Projects a vector (vec0) onto either another vector or a plane (default). If
    projecting onto another vector, vec1 defines that vector. If projecting onto
    a plane, vec1 defines a vector normal to the plane. Set Type='plane' for a
    plane, and Type='vector' for projection onto the vector. vec0 may be an array
    of vectors (Nx3). vec1 may be a single vector, or an array of vectors the
    same size as vec0
    
    vec=project(vec0,vec1,Type='normal')
    """
    
    vec0=np.atleast_2d(vec0)
    if np.ndim(vec1)==1:
        a=np.dot(vec0,vec1)
        para_vec=np.dot(np.transpose([a]),[vec1])
    elif np.ndim(vec1)==2 and np.shape(vec1)[0]==np.shape(vec0)[0]:
        a=np.multiply(vec0,vec1).sum(axis=1)
        para_vec=np.multiply(np.transpose([a]),vec1)
    else:
        print('vec1 must have one vector or the same number of vectors as vec0')
        return
    
    if Type.lower()[0]=='p':
        vec=vec0-para_vec
    elif Type.lower()[0]=='v':
        vec=para_vec
        
    return vec.T

def projectMOI(molecule,sel=None,select=None,**kwargs):
    
    mol=molecule
    
    sel1,sel2=selections(molecule,sel1=True,sel2=True,**kwargs)
    
    if sel is None:
        resi0=np.zeros(sel1.n_atoms,dtype=int)
        for k,s in enumerate(sel1):
            resi0[k]=s.resid
            
        sel=list()
        resi,index=np.unique(resi0,return_index=True) #all residues, and index for each selection
        for r in resi:
            sel.append(mol.mda_object.residues[r].atoms)
    else:
        sel=selections(molecule,sel0=sel,select=select)
        index=np.zeros(sel1.n_atoms)
        resi0=np.zeros(sel1.n_atoms,dtype=int)
        resi=np.zeros(1)

    def sub():
        vec=list()
        for s in sel:
            UT=s.principal_axes()
            I=s.moment_of_inertia()
            i=np.diag(UT.dot(I.dot(UT.T))).argmin()
            vec.append(UT[i])
            
        vec0=sel1.positions-sel2.positions
        
        vec_out=np.zeros(vec0.shape[-1::-1])
        
        for resi1,vec1 in zip(resi,vec):
            i=resi1==resi0
            vec_out[:,i]=project(vec0[i],vec1,Type='vector')
        
        return vec_out                
    return sub