#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:43:17 2019

@author: albertsmith
"""
import os
curdir=os.getcwd()
import numpy as np
os.chdir('../Struct')
import vf_tools as vft
os.chdir('../iRED')
from fast_index import trunc_t_axis
from Ct_fast import vec2data
from iRED_fast import vec2iRED
from fast_funs import get_trunc_vec
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

#%% Load data from a set of frames into multiple data objects
def frames2data(mol,frame_funs=None,frame_index=None,ffavg='direction',n=100,nr=10,label=None,avg_tensors=True,**kwargs):
    """
    Loads a set of frame functions into several data objects (the number of data
    objects produced is equal to the number of frames)
    
    Frames may be manually produced and included in a list (frame_funs), or 
    alternatively can be loaded into the molecule object itself (mol.new_frame)
    """
    
    
    
    vecs,avgs,tensors=get_vecs(mol,frame_funs,frame_index,ffavg,n,nr,**kwargs)
    if label is None and frame_funs is None:
        label=mol._frame_info['label']
    
    data=list()
    
    for q,(v,a,t) in enumerate(zip(vecs,avgs,tensors)):
        data.append(vec2data(v,molecule=mol,**kwargs))
        if avg_tensors:
            if frame_funs is None:frame_funs=mol._vf
            if frame_index is None:frame_index=mol._frame_info['frame_index']
            
            if q>0:
                "Extract the frame for the first point in the trajectory (index=np.array([0]))"
                v0=ini_vec_load(mol.mda_object.trajectory,[frame_funs[q-1]],[frame_index[q-1]],np.array([0]))['v'][0]
                if len(v0)==2:
                    v1,v2=v0
                else:
                    v1,v2=v0,None
                v1,inan=apply_index(v1,frame_index[q-1])
                v2,_=apply_index(v2,frame_index[q-1])
                
                sc=vft.getFrame(v1,v2)  #Angles to rotate first frame
                sc=[sc0.squeeze() for sc0 in sc]    

                "Apply to D2 evaluated at t=infty (D2inf)"                
                rho1=vft.pars2Spher(a['delta'],a['eta'],*a['euler'])
                rho=vft.Rspher(rho1,*sc)
                
                out=vft.Spher2pars(rho,return_angles=True)
                data[-1].vars['D2inf']={'delta':out[0],'eta':out[1],'euler':out[2:],'rho':a['rho']}
                
                "Apply to the averaged tensors"
                rho1=vft.pars2Spher(t['delta'],t['eta'],*t['euler'])
                rho=vft.Rspher(rho1,*sc)

                out=vft.Spher2pars(rho,return_angles=True)
                data[-1].vars['avg_tensors']={'delta':out[0],'eta':out[1],'euler':out[2:],'A0':t['rho']}
            else:
                data[-1].vars['D2inf']=a
                data[-1].vars['avg_tensors']=t
        if label is not None:
            data[-1].label=label
        
    return data

#%% Load data from a set of frames, and process with an iRED analysis
def frames2iRED(mol,frame_funs=None,frame_index=None,ffavg='direction',n=100,nr=10,\
                label=None,split_data=False,refVecs=False,**kwargs):
    """
    Loads a set of frame functions into a data object (or objects)
    
    Frames may be manually produced and included in a list (frame_funs), or 
    alternatively can be loaded into the molecule object itself (mol.new_frame)
    
    Currently, refVecs is only set to True or False, if True, mol.sel1 and 
    mol.sel2 will be used as reference vectors.
    """
    
    
        
    
    vecs=get_vecs(mol,frame_funs,frame_index,ffavg,n,nr,**kwargs)
    
    if refVecs:
        vf=mol._vf
        mol._vf=None
        kwargs['refVecs']=get_trunc_vec(mol,vecs[0]['index'])
        mol._vf=vf
        
    if label is None and frame_funs is None:
        label=mol._frame_info['label']
    
    if not(split_data):
        vec=vecs[0]
        vec['X']=np.concatenate([v['X'] for v in vecs],axis=1)
        vec['Y']=np.concatenate([v['Y'] for v in vecs],axis=1)
        vec['Z']=np.concatenate([v['Z'] for v in vecs],axis=1)
        data=vec2iRED(vec,molecule=mol,**kwargs)
        
        if label is not None:
            lbl=list()
            for k in range(len(vecs)):
                lbl.append(['f{0}_'.format(k)+'{0}'.format(l) for l in label])
                
            data.label=np.concatenate(lbl,axis=0)
    else:
        data=list()
        for v in vecs:
            data.append(vec2iRED(v,molecule=mol,**kwargs))
            if label is not None: data[-1].label=label
        
    return data
 
def get_vecs(mol,frame_funs=None,frame_index=None,ffavg='direction',n=100,nr=10,**kwargs):
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
    
    dt=[kwargs['dt'] if 'dt' in kwargs else traj.dt/1e3]
    
    vec0=ini_vec_load(traj,frame_funs,frame_index,index,dt)
    if ffavg=='eta':
        vecs,avgs,tensors=applyFrame2(vec0,return_avgs=True,tensor_avgs=True)
    else:
        vecs,avgs,tensors=applyFrame(vec0,ffavg,return_avgs=True,tensor_avgs=True)
    
    return vecs,avgs,tensors
    
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

#%% Apply the frames
def applyFrame(vecs,ffavg='direction',return_avgs=False,tensor_avgs=True):
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

    if tensor_avgs:
        "This is simply the averaged tensors"
        v1,v2=v0[-1]
        sc0=vft.getFrame(v1,v2)
        Davg=vft.D2(*sc0).mean(axis=-1) #Average of tensor components of frame 0
        out=vft.Spher2pars(Davg)    #Convert into delta,eta,euler angles
        tensors.append({'delta':out[0],'eta':out[1],'euler':out[2:]}) #Returns in a dict
        tensors[-1].update({'A0':Davg[2]})
    if return_avgs:
        """These are the expectation values for the correlation functions 
        of the individual components (m=-2->2) evaluated at infinite time
        """
        v1,_=v0[-1]
        Davg=vft.getD2inf(v1)
        sc0=[sc0.squeeze() for sc0 in vft.getFrame(v1[:,:,:1])]
        sc0[0]=1    #Set alpha to zero (not -gamma)
        sc0[1]=0
        A0=Davg[2]
        Davg=vft.Rspher(Davg,*sc0)
        out=vft.Spher2pars(Davg)
        avgs.append({'delta':out[0],'eta':out[1],'euler':out[2:]})
        avgs[-1].update({'A0':A0})
        
    "Next sweep from outer frame in"
    for k in range(nf-1):
        v1,v2=v0.pop(0) #Should have two vectors (or v1 and None)
        i=inan.pop(0)   #Here we remove motion from positions in this frame for NaN
        v1[0][i]=0
        v1[1][i]=0
        v1[2][i]=1
        if v2 is not None:
            v2[0][i]=1
            v2[1][i]=0
            v2[2][i]=0
        
        "Get the euler angles"
        sc=vft.getFrame(v1,v2) #Cosines and sines of alpha,beta,gamma (6 values ca,sa,cb,sb,cg,sg)
        
        "Apply rotations to all inner frames"
        vnew=list()
        for v in v0:
            v1,v2=v
            "Switch active to passive, apply rotation to all inner vectors"
            sci=vft.pass2act(*sc)
            vnew.append([vft.R(v1,*sci),vft.R(v2,*sci)])
        
        v0=vnew #Replace v0 for the next step
        
        "Apply the rotation matrix to some vector"
        if ffavg[0].lower()=='d' or ffavg[0].lower()=='f': #Direction or full (full not implemented yet)
            
            "Averaged tensors from last frame"
            if tensor_avgs:
                v1,v2=v0[-1]
                sc0=vft.getFrame(v1,v2)
                Davg=vft.D2(*sc0).mean(axis=-1) #Average of tensor components of frame 0
                out=vft.Spher2pars(Davg)    #Convert into delta,eta,euler angles
                tensors.append({'delta':out[0],'eta':out[1],'euler':out[2:]}) #Returns in a dict
                tensors[-1].update({'A0':Davg[2]})

            v1,_=v0[-1]
            Davg=vft.getD2inf(v1)
            A0=Davg[2]
            sc0=[sc0.squeeze() for sc0 in vft.getFrame(v1[:,:,:1])]
            sc0[0]=1    #Set alpha to 0 (as opposed to -gamma)
            sc0[1]=0
            Davg=vft.Rspher(Davg,*sc0)
            out=vft.Spher2pars(Davg)
            avgs.append({'delta':out[0],'eta':out[1],'euler':out[2:]})
            avgs[-1].update({'A0':A0})
            
            v=vft.R(np.array([0,0,1]),*avgs[-1]['euler'])   #Average of outer frame (usually bond)
        elif ffavg[0].lower()=='z':
            "z direction"
            v=np.atleast_2d([0,0,1]).T.repeat(nb,axis=1)
        elif ffavg[:3].lower()=='xyz':
            v=np.concatenate((np.array([[1,0,0]]).T.repeat(nb,axis=1),
                              np.array([[0,1,0]]).T.repeat(nb,axis=1),
                              np.array([[0,0,1]]).T.repeat(nb,axis=1)),axis=1)
            sc=[np.atleast_3d(s).repeat(3,axis=2).swapaxes(1,2).swapaxes(0,1).reshape([3*nb,nt]) for s in sc]
        
        v=vft.R(np.atleast_3d(v).repeat(nt,axis=2),*sc)    #Apply rotation of current frame to outer frame     
        vec_out.append({'X':v[0].T,'Y':v[1].T,'Z':v[2].T,'t':vecs['t'],'index':vecs['index']})
    
    if ffavg[:3].lower()=='xyz':
        v1,v2=v0.pop()
        sc=vft.getFrame(v1,v2) #Cosines and sines of alpha,beta,gamma (6 values ca,sa,cb,sb,cg,sg)
        v=np.concatenate((np.array([[1,0,0]]).T.repeat(nb,axis=1),
                          np.array([[0,1,0]]).T.repeat(nb,axis=1),
                          np.array([[0,0,1]]).T.repeat(nb,axis=1)),axis=1)
        sc=[np.atleast_3d(s).repeat(3,axis=2).swapaxes(1,2).swapaxes(0,1).reshape([3*nb,nt]) for s in sc]
        v=vft.R(np.atleast_3d(v).repeat(nt,axis=2),*sc)    #Apply rotation of current frame to outer frame
    else:
        v,_=v0.pop(0)
        l=np.sqrt(v[0]**2+v[1]**2+v[2]**2)
        v=v/l
            
    vec_out.append({'X':v[0].T,'Y':v[1].T,'Z':v[2].T,'t':vecs['t'],'index':vecs['index']})
    
    
    if return_avgs and tensor_avgs:
        for a in avgs:  #Convert to angles
            e=a['euler']
            alpha,beta,gamma=np.arctan2(e[1],e[0]),np.arctan2(e[3],e[2]),np.arctan2(e[5],e[4])
            a['euler']=np.concatenate(([alpha],[beta],[gamma]),axis=0)
        for a in tensors:  #Convert to angles
            e=a['euler']
            alpha,beta,gamma=np.arctan2(e[1],e[0]),np.arctan2(e[3],e[2]),np.arctan2(e[5],e[4])
            a['euler']=np.concatenate(([alpha],[beta],[gamma]),axis=0)
        return vec_out,avgs,tensors
    elif tensor_avgs:
        for a in tensors:  #Convert to angles
            e=a['euler']
            alpha,beta,gamma=np.arctan2(e[1],e[0]),np.arctan2(e[3],e[2]),np.arctan2(e[5],e[4])
            a['euler']=np.concatenate(([alpha],[beta],[gamma]),axis=0)
        return vec_out,tensors
    elif return_avgs:
        for a in avgs:  #Convert to angles
            e=a['euler']
            alpha,beta,gamma=np.arctan2(e[1],e[0]),np.arctan2(e[3],e[2]),np.arctan2(e[5],e[4])
            a['euler']=np.concatenate(([alpha],[beta],[gamma]),axis=0)
        return vec_out,avgs
    else:
        return vec_out

def applyFrame2(vecs,return_avgs=True,tensor_avgs=True):
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

    if tensor_avgs:
        "This is simply the averaged tensors"
        v1,v2=v0[-1]
        sc0=vft.getFrame(v1,v2)
        Davg=vft.D2(*sc0).mean(axis=-1) #Average of tensor components of frame 0
        out=vft.Spher2pars(Davg)    #Convert into delta,eta,euler angles
        tensors.append({'delta':out[0],'eta':out[1],'euler':out[2:]}) #Returns in a dict
        tensors[-1].update({'rho':Davg})
    if return_avgs:
        """These are the expectation values for the correlation functions 
        of the individual components (m=-2->2) evaluated at infinite time
        """
        v1,_=v0[-1]
        D2inf=vft.getD2inf(v1)
        sc0=[sc0.squeeze() for sc0 in vft.getFrame(v1[:,:,:1])]
        sc0[0]=1    #Set alpha to zero (not -gamma)
        sc0[1]=0
        A0=D2inf[2]
        D2inf=vft.Rspher(D2inf,*sc0)
        out=vft.Spher2pars(D2inf)
        avgs.append({'delta':out[0],'eta':out[1],'euler':out[2:]})
        avgs[-1].update({'rho':D2inf})
    
    "Next sweep from outer frame in"
    for k in range(nf-1):
        v1,v2=v0.pop(0) #Should have two vectors (or v1 and None)
        i=inan.pop(0)   #Here we remove motion from positions in this frame for NaN
        v1[0][i]=0
        v1[1][i]=0
        v1[2][i]=1
        if v2 is not None:
            v2[0][i]=1
            v2[1][i]=0
            v2[2][i]=0
        
        "Get the euler angles"
        sc=vft.getFrame(v1,v2) #Cosines and sines of alpha,beta,gamma (6 values ca,sa,cb,sb,cg,sg)
        
        "Apply rotations to all inner frames"
        vnew=list()
        for v in v0:
            v1,v2=v
            "Switch active to passive, apply rotation to all inner vectors"
            sci=vft.pass2act(*sc)
            vnew.append([vft.R(v1,*sci),vft.R(v2,*sci)])
        
        v0=vnew #Replace v0 for the next step
        


        "Averaged tensor from LAST frame"

        
        if tensor_avgs:
            v1,v2=v0[-1]
            sc0=vft.getFrame(v1,v2)
            Davg=vft.D2(*sc0).mean(axis=-1) #Average of tensor components of frame 0
            out=vft.Spher2pars(Davg)    #Convert into delta,eta,euler angles
            tensors.append({'delta':out[0],'eta':out[1],'euler':out[2:]}) #Returns in a dict
            tensors[-1].update({'rho':D2inf})

        v1,_=v0[-1]
        D2inf=vft.getD2inf(v1)
        sc0=[sc0.squeeze() for sc0 in vft.getFrame(v1[:,:,:1])]
        sc0[0]=1    #Set alpha to 0 (as opposed to -gamma)
        sc0[1]=0
        A0=D2inf[2]
        D2inf=vft.Rspher(D2inf,*sc0)
        out=vft.Spher2pars(D2inf)
        avgs.append({'delta':out[0],'eta':out[1],'euler':out[2:]})
        avgs[-1].update({'rho':D2inf})
        
        vZ=vft.R(np.array([0,0,1]),*avgs[-1]['euler'])   #Average of outer frame (usually bond)
        vZ=vft.R(np.atleast_3d(vZ).repeat(nt,axis=2),*sc)    #Apply rotation of current frame to outer frame
        vX=vft.R(np.array([1,0,0]),*avgs[-1]['euler'])   #Average of outer frame (usually bond)
        vX=vft.R(np.atleast_3d(vX).repeat(nt,axis=2),*sc)    #Apply rotation of current frame to outer frame
        vec_out.append({'X':{'X':vX[0].T,'Y':vX[1].T,'Z':vX[2].T},\
                        'Z':{'X':vZ[0].T,'Y':vZ[1].T,'Z':vZ[2].T},\
                             't':vecs['t'],'index':vecs['index'],'eta':avgs[-1]['eta']})
    
    "Apply to last frame"
    v,_=v0.pop(0)
    l=np.sqrt(v[0]**2+v[1]**2+v[2]**2)
    v=v/l
            
    vec_out.append({'X':v[0].T,'Y':v[1].T,'Z':v[2].T,'t':vecs['t'],'index':vecs['index']})
    
    if return_avgs and tensor_avgs:
        for a in avgs:  #Convert to angles
            e=a['euler']
            alpha,beta,gamma=np.arctan2(e[1],e[0]),np.arctan2(e[3],e[2]),np.arctan2(e[5],e[4])
            a['euler']=np.concatenate(([alpha],[beta],[gamma]),axis=0)
        for a in tensors:  #Convert to angles
            e=a['euler']
            alpha,beta,gamma=np.arctan2(e[1],e[0]),np.arctan2(e[3],e[2]),np.arctan2(e[5],e[4])
            a['euler']=np.concatenate(([alpha],[beta],[gamma]),axis=0)
        return vec_out,avgs,tensors
    elif tensor_avgs:
        for a in tensors:  #Convert to angles
            e=a['euler']
            alpha,beta,gamma=np.arctan2(e[1],e[0]),np.arctan2(e[3],e[2]),np.arctan2(e[5],e[4])
            a['euler']=np.concatenate(([alpha],[beta],[gamma]),axis=0)
        return vec_out,tensors
    elif return_avgs:
        for a in avgs:  #Convert to angles
            e=a['euler']
            alpha,beta,gamma=np.arctan2(e[1],e[0]),np.arctan2(e[3],e[2]),np.arctan2(e[5],e[4])
            a['euler']=np.concatenate(([alpha],[beta],[gamma]),axis=0)
        return vec_out,avgs
    else:
        return vec_out

    
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

#%% Progress bar for loading/aligning
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()