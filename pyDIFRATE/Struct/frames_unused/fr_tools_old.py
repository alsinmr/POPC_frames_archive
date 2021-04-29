#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:29:06 2020

@author: albertsmith
"""

import os
curdir=os.getcwd()
import numpy as np
os.chdir('../Struct')
import vf_tools as vft
os.chdir('../iRED')
from fast_index import trunc_t_axis
from iRED_fast import vec2iRED
from fast_funs import get_trunc_vec,get_count,printProgressBar
os.chdir('../data')
from data_class import data
os.chdir(curdir)

"""
The frame module is one of the more complex components of pyDIFRATE. It's worth
noting the structure of some of the data elements here, for future reference.

First, before executing the functions here, we need one or more frame functions.
Typically, these are stored as part of the molecule object. mol._vf is a list of
functions, where the length of the list is the number of frames. Each frame function
returns either a 3xN matrix, or a list with two elements, each being a 3xN matrix.
N is the number of sub-frames (for example, if using the "bond" frame, then there
is a sub-frame for each bond). 3 is the number of spatial dimensions.

When we run ini_vec_load, we loop over the trajectory and execute all frame 
functions at each time point. This returns a dictionary object, where the element
'v' contains the results of this.
v is a list of lists. The outer list runs over the total number of frames in use,
and the inner list runs over the number of time points in the trajectory that
are used (other elements of the dict keep track of which time points are used).

Here an important note: each frame function may return matrices of different 
sizes. First, we may apply the same sub-frame of one frame to multiple frames of
another frame. For example, our last frame is usually the "bond" frame, but we
could have a "MOIz" frame as the outer frame, which would be the same for all
bonds in part of a molecule. Then, we will have more sub-frames in the "bond" 
frame than in the "MOIz" frame. This needs to be fixed in a latter step. The 
second important point: some frames are defined by a single vector, and some
frames by two vectors. For example, the "bond" frame is simply defined by a 
single vector (direction of the bond), but the "superimpose" frame is described
by two vectors. 

Size of v: [# of frames, # of time points, []]
where [] can have variable size/number of dimensions
size of []: [2,# of dimensions(=3), # of sub-frames]  in case of two vectors
or      []: [# of dimensions (=3), # of sub-frames]

Within ini_vec_load, we swap around the dimensions, such that
v: [# of frames,[]]
[]:  if 1 vector [# of dimension(=3), # of sub-frames, # of time points]        
[]: if 2 vectors [2, # of dimension(=3), # of sub-frames, # of time points]

In the latter steps, we need to make sure all the matrix dimensions are the same.
This is achieved, first by filling in None where only one vector describes the
frame (occuring within applyFrame function)

Size of v0: [# of frames,2]
[]:  None  -- or --      
[]: [ # of dimension(=3), # of sub-frames, # of time points]

Within applyFrame, the apply_index function is run. In this case, all frames are
indexed so that the inner dimensions have the same size. frame_index is usually
stored in mol._frame_info['frame_index']. It should be a list of length equal
to the number of frames, and the sublists should be 1D arrays with equal 
number of elements. The values determine which sub-frame is applied. For some
frames, usually this will just be an index from 0-# of sub-frames (for example,
the "bond" frame should almost always be indexed from 0-# of bonds). Note, 
individual elements of the frame_index may be set to None (nan in numpy), which
will yield a nan vector to be returned. 

After running apply_index, 

Size of v: [# of frames, 2,[]]
[]: None -- or --
[]: [# of dimensions (=3), # of bonds, # of time points]
(by bonds, effectively we mean the largest number of sub-frames)


We then loop over each frame. For each frame, we acquire the cos/sin of the
euler angles (if one vector provided, then alpha=-gamma, else we get 3 
independent euler angles). These angle are then applied in reverse to all 
inner frames (this would rotate the current frame to z, with the second vector, 
if given, into the xz-plane). In the inner loops, we continue this process, where
the vectors rotated by the outer frame are used. 

At each frame, we use the acquired angles to apply to some vector. This vector
could just be a vector along z (then rotated at each time point), but we often
will apply to the vector pointing the same direction as the tensor average of
the innermost frame (thus obtaining the influence of the outer frame on the
detector responses/relaxation)

The results are stored in a list of dictionaries. The list is the same length
as the number of frames. In each dictionary, 'X','Y', and 'Z' are stored for
the three dimensions. Then, the stored vectors in 'X','Y','Z' each have size 
# of subframes, # of time points

"""

"""
For tomorrow, we need to make sure the correct averages are coming into the 
calculation of the correlation function.
"""

#%% Load data from a set of frames into multiple data objects
def frames2data(mol,frame_funs=None,frame_index=None,mode='full',n=100,nr=10,label=None,dt=None):
    """
    Loads a set of frame functions into several data objects (the number of data
    objects produced is equal to the number of frames)
    
    Frames may be manually produced and included in a list (frame_funs), or 
    alternatively can be loaded into the molecule object itself (mol.new_frame)
    """
    
    
    if mode.lower()=='both':
        vecs,vecs_z,avgs,tensors=get_vecs(mol,frame_funs=frame_funs,frame_index=frame_index,mode=mode,n=n,nr=nr,dt=dt)
    else:
        vecs,avgs,tensors=get_vecs(mol,frame_funs=frame_funs,frame_index=frame_index,mode=mode,n=n,nr=nr,dt=dt)
    if label is None and frame_funs is None:
        label=mol._frame_info['label']
    
    data=list()
    
    "Here, we get the orientation of all frames at t=0"
    if frame_funs is None:frame_funs=mol._vf
    if frame_index is None:frame_index=mol._frame_info['frame_index']
    out=ini_vec_load(mol.mda_object.trajectory,frame_funs,frame_index,np.array([0]))

    for q,(v,a,t) in enumerate(zip(vecs,avgs,tensors)):
        if mode.lower()=='both':
            data.append([vec2data(v,a,mode='full',molecule=mol),\
                         vec2data(vecs_z[q],mode='z',molecule=mol)])
        else:
            data.append(vec2data(v,a,mode=mode,molecule=mol))

        
        if q>0:
            v0=out['v'][q-1]
            v1,v2=v0 if len(v0)==2 else (v0,None)
            
            v1,inan=apply_index(v1,frame_index[q-1])
            v2,_=apply_index(v2,frame_index[q-1])
            
            sc1=vft.getFrame(v1,v2)
            sc1=[sc0.squeeze() for sc0 in sc1]
            
            rho=vft.Rspher(t['rho'],*sc1)
            new=vft.Spher2pars(rho,return_angles=True)
            avg_tensors={'delta':new[0],'eta':new[1],'euler':new[2:],'rho':t['rho']}          
        else:
            avg_tensors=t

        out1=out.copy()
        i=np.concatenate((np.arange(q),[len(vecs)-1]))
        out1['v']=[out1['v'][k] for k in i]
        out1['frame_index']=[out1['frame_index'][k] for k in i]
        vecs1,_,_=applyFrame(out1)
        sc=[vft.getFrame(v['vZ'],v['vX']) for v in vecs1[:-1]]
        sc.append(vft.getFrame(vecs1[-1]['vZ']))
        
        rho=a['rho']
        for sc0 in sc[::-1]:
            rho=vft.Rspher(rho,*sc0)

        new=vft.Spher2pars(rho.squeeze(),return_angles=True)
        
        D2inf={'delta':new[0],'eta':new[1],'euler':new[2:],'rho':a['rho']}
        
        if mode.lower()=='both':
            if label is not None:
                data[-1][0].label=label
                data[-1][1].label=label
            data[-1][0].vars['D2inf']=D2inf
            data[-1][1].vars['D2inf']=D2inf
            data[-1][0].vars['avg_tensors']=avg_tensors
            data[-1][1].vars['avg_tensors']=avg_tensors
        else:
            if label is not None:
                data[-1].label=label
            data[-1].vars['D2inf']=D2inf
            data[-1].vars['avg_tensors']=avg_tensors
        
    if mode.lower()=='both':
        data=[[d[0] for d in data],[d[1] for d in data]]
    return data


def frames2Ct(mol,mode='separate',n=10,nr=10,dt=None):
    """
    Sets up calculation of a correlation function from a molecule object. In 
    'full' mode or 'P2' mode, this will simply return the correlation functions
    otherwise found in a data object (use frames2data). However, if the separated
    correlation functions are desired, then one must use this function for setup
    since the array dimensionality is incompatible with the 'data' object.
    
    ct,avgs,tensors = frames2Ct(mol,mode='separate',n=10,nr=10,dt=None)
    
        or for just the correlation functions (no frame index will be applied)
    
    ct=frames2Ct(frame_funs,mode='separate',n=10,nr=10,dt=None)
    """
    
    if dt is None:dt=mol.mda_object.trajectory.dt/1e3
    index=trunc_t_axis(mol.mda_object.trajectory.n_frames,n,nr)
    

    vecs,avgs,tensors=get_vecs(mol=mol,frame_funs=mol._vf,frame_index=mol._frame_info['frame_index'],\
                               mode=mode,n=n,nr=nr,dt=dt)
        
    ct=list()
    
    for v in vecs:
        if 'vX' in v.keys():
            ct.append(Ct(vZ=v['vZ'],vX=v['vX'],vY=v['vY'],mode='separate',index=index,avgs=None,dt=dt))
        else:
            ct.append(Ct(vZ=v['vZ'],index=index,dt=dt))
    
    return ct,avgs,tensors

#%% Load in vectors for each frame
def ini_vec_load(traj,frame_funs,frame_index=None,index=None,dt=None):
    """
    Loads vectors corresponding to each frame, defined in a list of frame functions.
    Each element of frame_funs should be a function, which returns one or two
    vectors.
    
    traj should be the trajectory iterable

    index (optional) is used for sparse sampling of the trajectory
    
    dt gives the step size of the MD trajectory (for correcting incorrect step sizes in trajectory)
    """
    
    if hasattr(frame_funs,'__call__'):frame_funs=[frame_funs]  #In case only one frame defined (unusual usage)

    nf=len(frame_funs)
    nt=traj.n_frames
    
    if index is None: index=np.arange(nt)
    if dt is None: dt=traj.dt
    
    t=index*dt
    v=[list() for _ in range(nf)]
    """v is a list of lists. The outer list runs over the number of frames (length of frame_funs)
    The inner list runs over the timesteps of the trajectory (that is, the timesteps in index)
    The inner list contains the results of executing the frame function (outer list) at that
    time point (inner list)
    """
      
    for c,i in enumerate(index):
        traj[i] #Go to current frame
        for k,f in enumerate(frame_funs):
            v[k].append(f())
        "Print the progress"
        try:
            if c%int(len(index)/100)==0 or c+1==len(index):
                printProgressBar(c+1, len(index), prefix = 'Loading Ref. Frames:', suffix = 'Complete', length = 50) 
        except:
            pass
                    
    
    SZ=list()        
    
    for k,v0 in enumerate(v):
        v[k]=np.array(v0)
        """Put the vectors in order such that if two vectors given, each vector
        is an element of the first dimension, and the second dimension is X,Y,Z
        (If only one vector, X,Y,Z is the first dimension)
        """
        if v[k].ndim==4:
            v[k]=((v[k].swapaxes(0,1)).swapaxes(1,2)).swapaxes(2,3)
        else:
            v[k]=(v[k].swapaxes(0,1)).swapaxes(1,2)
        SZ.append(v[k].shape[-2])
        
    SZ=np.array(SZ)
    SZ=SZ[SZ!=1]

    "Below line causing problems- comment for the moment...means frame_index must be given elsewhere..."    
#    if np.all(SZ==SZ[0]):
#        frame_index=np.repeat([np.arange(SZ[0])],nf,axis=0)    
    
    "Somehow, the above line is required for proper iRED functioning...very strange"
    "!!!! Fix and understand above line's glitch!!!!"
    
    return {'n_frames':nf,'v':v,'t':t,'index':index,'frame_index':frame_index} 


 
def get_vecs(mol,frame_funs=None,frame_index=None,mode='full',n=100,nr=10,dt=None):
    "Get the set of frame functions"
    if frame_funs is None:
        if mol._vf is not None:
            frame_funs=mol._vf
            if frame_index is None:
                frame_index=mol._frame_info['frame_index']
        else:
            print('No frame functions provided')
            return
    
    traj=mol.mda_object.trajectory
    index=trunc_t_axis(traj.n_frames,n,nr)
    
    if dt is None:dt=traj.dt/1e3
    
    vec0=ini_vec_load(traj,frame_funs,frame_index,index,dt)

    if mode.lower()=='both':
        vecs,avgs,tensors=applyFrame(vec0,mode='full')
        vecs_z,_,_=applyFrame(vec0,mode='z')
        
        return vecs,vecs_z,avgs,tensors
    else:
        vecs,avgs,tensors=applyFrame(vec0)
        
        return vecs,avgs,tensors

def vec2data(vec,avgs=None,mode='full',**kwargs):
    """
    Takes a vector and creates the corresponding data object
    
    data=vec2data(vec,**kwargs)
    """
    
    dt=(vec['t'][1]-vec['t'][0])/(vec['index'][1]-vec['index'][0])
   
    if 'vX' in vec.keys():
        ct=Ct(vZ=vec['vZ'],index=vec['index'],dt=dt,vX=vec['vX'],vY=vec['vY'],avgs=avgs,mode=mode)
    else:
        ct=Ct(vZ=vec['vZ'],index=vec['index'],dt=dt)
    Ctdata=data(Ct=ct,**kwargs)
    
    return Ctdata

#%% Apply the frames
def applyFrame(vecs,mode='full'):
    """
    Calculates vectors, which may then be used to determine the influence of
    each frame on the overall correlation function (via detector analysis)
    
    For each frame, we will calculate its trajectory with respect to the previous
    frame (for example, usually the first frame should be some overall motion
    and the last frame should be the bonds themselves. The overall motion is 
    calculated w.r.t. the lab frame)
    
    ffavg is "Faster-frame averaging", which determines how we take into account
    the influence of outer frames on the inner frames. 
     ffavg='off' :      Neglect influence of outer frames, only calculate behavior
                        of frames with respect to the previous frame
     ffavg='direction': Apply motion of a given frame onto a vector pointing
                        in the same direction as the tensor obtained by averaging
                        the tensor in the outer frame
     ffavg='full' :     Apply motion using direction, and also scale
                        (currently unavailable...)
    
    """
    
    v0=vecs['v']
    nf=len(v0)
    nt=vecs['t'].size
    
    avgs=list()
    
    fi=vecs['frame_index']
    nb=fi[0].size
    
    "Make sure all vectors in v0 have two elements"
    v0=[v if len(v)==2 else [v,None] for v in v0]
    
    "Start by updating vectors with the frame index"
#    v0=[[apply_index(v0[k][0],fi[k]),apply_index(v0[k][1],fi[k])] for k in range(nf)]
    inan=list()
    v1=list()
    for k in range(nf):
        a,b=apply_index(v0[k][0],fi[k])
        c,_=apply_index(v0[k][1],fi[k])
        inan.append(b)
        v1.append([a,c])
    v0=v1


    
    vec_out=list()
    avgs=list()
    tensors=list()


    "This is simply the averaged tensors– Not used, just for storage and later plotting"
    v1,v2=v0[-1]
    sc0=vft.getFrame(v1,v2)
    Davg=vft.D2(*sc0).mean(axis=-1) #Average of tensor components of frame 0
    out=vft.Spher2pars(Davg)    #Convert into delta,eta,euler angles
    tensors.append({'delta':out[0],'eta':out[1],'euler':out[2:]}) #Returns in a dict
    tensors[-1].update({'A0':Davg[2]})

    """These are the expectation values for the correlation functions 
    of the individual components (m=-2->2) evaluated at infinite time
    """
    v1,_=v0[-1]     #This is the bond direction in the lab frame
    Davg=vft.getD2inf(v1)
    out=vft.Spher2pars(Davg)
    avgs.append({'delta':out[0],'eta':out[1],'euler':out[2:]})
    avgs[-1].update({'rho':Davg})
        
    
    "Next sweep from outer frame in"
    for k in range(nf-1):
        v1,v2=v0.pop(0) #Here we import the outermost frame (that is, the slowest frame)
        i=inan.pop(0)   #Here we remove motion from positions in this frame for NaN
        v1[0][i]=0      #Frames with NaN here point along z for all times
        v1[1][i]=0
        v1[2][i]=1
        if v2 is not None:  #Frames with NaN here point along x for all times
            v2[0][i]=1
            v2[1][i]=0
            v2[2][i]=0
        
        "Get the euler angles of the outermost frame"
        sc=vft.getFrame(v1,v2) #Cosines and sines of alpha,beta,gamma (6 values ca,sa,cb,sb,cg,sg)
        
        """Apply rotations to all inner frames- that is, all inner frames are given in the outer frame
        Keep in mind that averaged tensors inside this frame are no longer expressed in the lab frame"
        This is later taken care of in 
        """
        vnew=list()
        for v in v0:
            v1,v2=v
            "Switch active to passive, apply rotation to all inner vectors"
            sci=vft.pass2act(*sc)
            vnew.append([vft.R(v1,*sci),vft.R(v2,*sci)])
        
        v0=vnew #Replace v0 for the next step
        
        
        "Averaged tensors from last frame- Not using this, just for record keeping"
        v1,_=v0[-1] #This is the bond direction in the current frame
        sc0=vft.getFrame(v1)
        Davg=vft.D2(*sc0).mean(axis=-1) #Average of tensor components of frame 0
        out=vft.Spher2pars(Davg)    #Convert into delta,eta,euler angles
        tensors.append({'delta':out[0],'eta':out[1],'euler':out[2:]}) #Returns in a dict
        tensors[-1].update({'rho':Davg})

        "D2 evaluated at infinite time of last frame"
        D2inf=vft.getD2inf(v1)   #Average D2 components in current frame at infinite time
        out=vft.Spher2pars(D2inf)
        
        avgs.append({'delta':out[0],'eta':out[1],'euler':out[2:]})
        avgs[-1].update({'rho':D2inf})
        
        if mode[0].lower()=='z':
            vZ=vft.R(np.array([0,0,1]),*avgs[-1]['euler'])   #Average of outer frame (usually bond)
            vZ=vft.R(np.atleast_3d(vZ).repeat(nt,axis=2),*sc)    #Apply rotation of current frame to outer frame
            vZ=vZ.swapaxes(1,2)
            vec_out.append({'vZ':vZ,'t':vecs['t'],'index':vecs['index']})
        else:        
            vX=vft.R(np.array([1,0,0]),*sc)
            vY=vft.R(np.array([0,1,0]),*sc)
            vZ=vft.R(np.array([0,0,1]),*sc)
            vX=vX.swapaxes(1,2)
            vY=vY.swapaxes(1,2)
            vZ=vZ.swapaxes(1,2)
            
            vec_out.append({'vZ':vZ,'vX':vX,'vY':vY,'t':vecs['t'],'index':vecs['index']})

    v,_=v0.pop(0)   #Operations on all but bond frame result in normalization, but here we have to enforce length
    l=np.sqrt(v[0]**2+v[1]**2+v[2]**2)
    v=v/l
    
    v=v.swapaxes(1,2)
            
    vec_out.append({'vZ':v,'t':vecs['t'],'index':vecs['index']})
    
    return vec_out,avgs,tensors

def Ct(vZ,dt,index=None,vX=None,vY=None,avgs=None,mode='full'):
    """
    Calculates the correlation function of a motion given the vX and vZ 
    vectors, in addition to the "avgs" dict, which contains components of the
    averaged elements of D2 of the inner (bond) frame, evaluated in the current
    frame.
    
    Default mode is 'full', which yields the correlation function considering the
    residual tensor from a prior motion (tensor normalized by the 0 component).
    
    This correlation function is
    
    C(t)*A_0 
      = <D^2_00>A_0+Re(<D^2_10>+<D^2_-10>)Re(A_1)-Im(<D^2_10>+<D^2_-10>)Im(A_1)
    
    where the A_p are the averaged components from a prior motion
    
    If the mode is set to "separate" or just "s", then the components of the above
    correlation are returned in a 5xNresxNt array (in the Ct field of the output
    dictionary)
    
    <D^2_00>, to be multiplied by A_0 
    Re(<D^2_10>-<D^2_-10>), to be multiplied by Re(A_1)
    -Im(<D^2_10>+<D^2_-10>), to be multiplied by Im(A_1)
    Re(<D^2_20>+<D^2_-20>), to be multiplied by Re(A_2)
    Re(<D^2_20>+<D^2_-20>), to be multiplied by Im(A_2)
    
    If 'avgs' is provided in this case, then the terms will be multiplied by
    the appropriate element from 'avgs'. Otherwise, the component is returned
    without this multiplication.
    
    Finally, one may select simply 'P2' (for the second Legendre polynomial),
    in which case <D^2_00> is returned. If 'avgs' is not given, and mode is not
    set to 's', then this will be the default option
    

    out = Ct(vZ,index,dt,vX=None,vY=None,avgs=None,mode='full')
    
    out contains keys 't', 'Ct', 'index', and 'N'
    """
    
    n=np.size(index)

    
    if mode[0].lower()=='s':
        if avgs is None:
            rho=np.array([1+1j,1+1j,1,1+1j,1+1j])
        else:
            rho=avgs['rho']/avgs['rho'][2].real
        c=[np.zeros([np.max(index)+1,np.shape(vZ[0])[1]]) for k in range(5)]
        for k in range(n):
            CaSb=vX[0,k]*vZ[0,k:]+vX[1,k]*vZ[1,k:]+vX[2,k]*vZ[2,k:]
            SaSb=vY[0,k]*vZ[0,k:]+vY[1,k]*vZ[1,k:]+vY[2,k]*vZ[2,k:]
            Cb=vZ[0,k]*vZ[0,k:]+vZ[1,k]*vZ[1,k:]+vZ[2,k]*vZ[2,k:]
            
            c[0][index[k:]-index[k]]+=(-1/2+3/2*Cb**2)*rho[2].real
            c[1][index[k:]-index[k]]+=np.sqrt(3/2)*(CaSb*Cb)*rho[3].real
            c[2][index[k:]-index[k]]+=np.sqrt(3/2)*(SaSb*Cb)*rho[3].imag
            c[3][index[k:]-index[k]]+=np.sqrt(3/8)*(CaSb**2-SaSb**2)*rho[4].real
            c[4][index[k:]-index[k]]+=np.sqrt(3/2)*(CaSb*SaSb)*rho[4].imag
            
            if k%np.ceil(n/100).astype(int)==0 or k+1==n:
                printProgressBar(k+1, len(index), prefix = 'Calculating C(t):', suffix = 'Complete', length = 50) 
        
        N=get_count(index)  #Number of averages for each time point
        i=N!=0              #Non-zero number of averages
        N=N[i]              #Remove all zeros
    
        ct0=np.array([c0[i].T/N for c0 in c])
        if avgs is not None:
            ct=ct0.sum(axis=0)
        else:
            ct=None
    else:
        c=np.zeros([np.max(index)+1,np.shape(vZ[0])[1]])
        if avgs is None or mode[:2].lower()=='p2':
            for k in range(n):
                Cb2=(vZ[0][k:]*vZ[0][k]+vZ[1][k:]*vZ[1][k]+vZ[2][k:]*vZ[2][k])**2  #Cosine beta^2
                c[index[k:]-index[k]]+=-1/2+3/2*Cb2
        else:
            if vX is None or vY is None:
                print('vX and vY required for "full" mode')
                return
            rho=avgs['rho']/avgs['rho'][2].real
            
            for k in range(n):            
                CaSb=vX[0,k]*vZ[0,k:]+vX[1,k]*vZ[1,k:]+vX[2,k]*vZ[2,k:]
                SaSb=vY[0,k]*vZ[0,k:]+vY[1,k]*vZ[1,k:]+vY[2,k]*vZ[2,k:]
                Cb=vZ[0,k]*vZ[0,k:]+vZ[1,k]*vZ[1,k:]+vZ[2,k]*vZ[2,k:]
                
                c[index[k:]-index[k]]+=(-1/2+3/2*Cb**2)*rho[2].real+\
                                        np.sqrt(3/2)*(CaSb*Cb)*rho[3].real+\
                                        np.sqrt(3/2)*(SaSb*Cb)*rho[3].imag+\
                                        np.sqrt(3/8)*(CaSb**2-SaSb**2)*rho[4].real+\
                                        np.sqrt(3/2)*(CaSb*SaSb)*rho[4].imag
                if k%np.ceil(n/100).astype(int)==0 or k+1==n:
                    printProgressBar(k+1, len(index), prefix = 'Calculating C(t)', suffix = 'Complete', length = 50) 
        
        N=get_count(index)  #Number of averages for each time point
        i=N!=0              #Non-zero number of averages
        N=N[i]              #Remove all zeros
    
        ct=c[i].T/N         #Remove zeros, normalize
        ct0=None
        
        
    t=np.linspace(0,dt*np.max(index),index[-1]+1)
    t=t[i]
    
    Ct={'t':t,'N':N,'index':index}
    if ct is not None:
        Ct['Ct']=ct
    if ct0 is not None:
        Ct['<D2>']=ct0
        
    
    return Ct

def Ct_ft(vZ,vX=None,vY=None,D2='D20'):
    """
    Calculates a correlation function (or functions) using the Fourier Transform.
    One may select which correlation function to return, with the options being
    
    D2=
    
    D20:    <D^2_00>, to be multiplied by A_0 
    ReD21:  Re(<D^2_10>+<D^2_-10>), to be multiplied by Re(A_1)
    ImD21:  -Im(<D^2_10>+<D^2_-10>), to be multiplied by Im(A_1)
    ReD22:  Re(<D^2_20>-<D^2_-20>), to be multiplied by Re(A_2)
    ImD22:  Re(<D^2_20>+<D^2_-20>), to be multiplied by Im(A_2)
    all:    Returns all 5 functions in the order listed above
    
    Ct_ft(vZ,vX=None,vY=None,D2='D20'):
        
    vX and vY are required for calculating all but D20
    
    out = Ct_ft(vZ,vX=None,vY=None,D2='D20'):
    """


    
    
    if (vX is None or vY is None) and D2.lower()!='d20':
        print('vX and vY must be provided to calculate for D2={0}'.format(D2))
        return
    

    ZZ=list()
    for k in range(3):  #Loop over x,y,z for first vector
        for j in range(3):  #Loop over x,y,z for second vector
            ZZ.append(fft_sym(vZ[k]*vZ[j]))
                
    if D2.lower()=='all' or D2.lower()=='red21':
        XZ=list()
        YZ=list()
        for k in range(3):
            for j in range(3):
                XZ.append(fft_sym(vX[k]*vZ[j]))
    
    if D2.lower()=='all' or D2.lower()=='imd21':
        YZ=list()
        for k in range(3):
            for j in range(3):
                YZ.append(fft_sym(vY[k]*vZ[j]))
        
            
    if D2.lower()=='all' or D2.lower()=='red22':
        XX=list()
        YY=list()
        for k in range(3):
            for j in range(3):
                XX.append(fft_sym(vX[k]*vX[j]))
                YY.append(fft_sym(vY[k]*vY[j]))
                
    if D2.lower()=='all' or D2.lower()=='imd22':
        XY=list()
        for k in range(3):
            for j in range(3):
                XY.append(fft_sym(vX[k]*vY[j]))
   
    if D2.lower()=='all':
        D2=['D20','ReD21','ImD21','ReD22','ImD22']
    else:
        D2=[D2]
    
    Ct=list()

    for d2 in D2:
        if d2.lower()=='d20':
            Ct.append(-1/2+3/2*fft_sym(np.sum([zz*zz.conj() for zz in ZZ],axis=0),inv=True))
        elif d2.lower()=='red21':
#            Ct.append(np.sqrt(3/2)*fft_sym(np.sum([0.5*(xz*zz.conj()+xz.conj()*zz) for xz,zz in zip(XZ,ZZ)],axis=0),inv=True))
            Ct.append(np.sqrt(3/2)*fft_sym(np.sum([xz*zz.conj() for xz,zz in zip(XZ,ZZ)],axis=0),inv=True))
        elif d2.lower()=='imd21':
#            Ct.append(np.sqrt(3/2)*fft_sym(np.sum([0.5*(yz*zz.conj()+yz.conj()*zz) for yz,zz in zip(YZ,ZZ)],axis=0),inv=True))
            Ct.append(np.sqrt(3/2)*fft_sym(np.sum([yz*zz.conj() for yz,zz in zip(YZ,ZZ)],axis=0),inv=True))
        elif d2.lower()=='red22':
#            Ct.append(np.sqrt(3/8)*fft_sym(np.sum([0.5*(xx*zz.conj()+xx.conj()*zz-yy*zz.conj()-yy.conj()*zz) for xx,yy,zz in zip(XX,YY,ZZ)],axis=0),inv=True))
            Ct.append(np.sqrt(3/8)*fft_sym(np.sum([xx*zz.conj()-yy*zz.conj() for xx,yy,zz in zip(XX,YY,ZZ)],axis=0),inv=True))
        elif d2.lower()=='imd22':
#            Ct.append(np.sqrt(3/2)*fft_sym(np.sum([0.5*(xy*zz.conj()+xy.conj()+zz) for xy,zz in zip(XY,ZZ)],axis=0),inv=True))
            Ct.append(np.sqrt(3/2)*fft_sym(np.sum([xy*zz.conj() for xy,zz in zip(XY,ZZ)],axis=0),inv=True))
        
    if len(Ct)==1:Ct=Ct[0]
    
    return Ct

def fft_sym(x,inv=False):
    """
    Performs an fft after symmetrizing the input. Also makes sure data is 2D. Set
    inv=True in order to take the inverse Fourier transform, and truncate 
    """
    
    if inv:
        X=np.atleast_2d(x)
        x=np.fft.ifft(X,axis=1)
        x=x[:,:int(X.shape[1]/2)]
        return x.real/X.shape[1]
    else:
        x=np.atleast_2d(x)
        x=np.concatenate([x,x[:,::-1]],axis=1)
        X=np.fft.fft(x,axis=1)
        return X
    

def apply_index(v0,index):
    """
    Applies the frame index to a set of variables (the frame index is applied
    to the second dimension)
    """
    
    if v0 is None:
        return None,None
    
    "SZ : [# of dimesions (=3), # of bonds, # of time points]"
    SZ=[np.shape(v0)[0],np.size(index),np.shape(v0)[2]]
    vout=np.zeros(SZ)
    
#    if np.any(np.isnan(index)):
    inan=np.isnan(index)    #Keep track of where nan are
    i=index.copy()  #Don't edit the original!
    i[inan]=0   #Just set to the first frame in the index
#    else:
#        i=index
#        inan=None
    
    for k,v in enumerate(v0):       #Looping over the dimensions
        vout[k]=v[np.array(i).astype(int)]
#        if inan is not None:
#            vout[k][inan]=1 if k==2 else 0
    return vout,inan

