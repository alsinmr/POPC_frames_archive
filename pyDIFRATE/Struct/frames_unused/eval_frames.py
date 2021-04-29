#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:47:41 2020

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



def loop_gen(vZ,vXZ=None,nuZ_f=None,nuXZ_f=None,nuZ_F=None,nuXZ_F=None,m=None):
    """
    Generates the loop items required for calculating correlation functions and
    equilibrium values.
    
    For f in F calculations, vZ and nuZ_f are required, with additional optional
    arguments nuXZ_f, nuZ_F (calc in LF if omitted), nuXZ_F. If nuZ_f is included,
    a dictionary with the following keys is returned for each of 81 iterator 
    elements
    
    Dictionary keys:
        eag:    "gamma" element of the vector representing the "alpha" axis for
                projection from frame f into frame F
        ebd:    "delta" element of the vector representing the "beta" axis for
                projection from frame f into frame F
        gz:     "gamma" element of the bond vector in frame f
        dz:     "delta" element of the bond vector in frame f
        ax:     "alpha" element of the x-axis of the bond axis system in frame F
        ay:     "alpha" element of the y-axis of the bond axis system in frame F
        az:     "alpha" element of the z-axis of the bond axis system in frame F
        bx:     "beta" element of the x-axis of the bond axis system in frame F
        by:     "beta" element of the y-axis of the bond axis system in frame F
        bz:     "beta" element of the z-axis of the bond axis system in frame F
        
    For PAS calculations (optionally in frame F), vZ is required, with additional
    optional arguments nuZ_F (calc in LF if omitted) and nuXZ_F. In this case,
    a dictionary is returned with the following keys for each of 9 iterator 
    elements
    
    Dictionary keys:
        ax:     "alpha" element of the x-axis of the bond axis system
        ay:     "alpha" element of the y-axis of the bond axis system
        az:     "alpha" element of the z-axis of the bond axis system
        bx:     "beta" element of the x-axis of the bond axis system
        by:     "beta" element of the y-axis of the bond axis system
        bz:     "beta" element of the z-axis of the bond axis system
    
    If m is provided, only the dictionary elements required for calculating the mth
    component will be returned
    
    loop_gen(vZ,nuZ_f=None,nuXZ_f=None,nuZ_F=None,nuXZ_F=None,m=None)
    """
    
    "Apply frame F"
    vZF,vXZF,nuZ_fF,nuXZ_fF=applyFrame(vZ,vXZ,nuZ_f,nuXZ_f,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F)
    
    if m!=0: 
        sc=vft.getFrame(vZF,vXZF)
        vXF=vft.R([1,0,0],*sc)  #x and y axes of the bond axes in F
        vYF=vft.R([0,1,0],*sc)

    if nuZ_f is None:
        if m is None or m==-2 or m==2:
            for ax,ay,az in zip(vXF,vYF,vZF):
                for bx,by,bz in zip(vXF,vYF,vZF):
                    out={'ax':ax,'ay':ay,'az':az,'bx':bx,'by':by,'bz':bz}
                    yield out
        elif m==-1 or m==1:
            for ax,ay,az in zip(vXF,vYF,vZF):
                for bz in vZF:
                    out={'ax':ax,'ay':ay,'az':az,'bz':bz}
                    yield out
        elif m==0:
            for az in vZF:
                for bz in vZF:
                    out={'az':az,'bz':bz}
                    yield out
                    
    else:
        sc=vft.getFrame(nuZ_fF,nuXZ_fF)
        vZf=vft.R(vZF,*vft.pass2act(*sc))
        eFf=[vft.R([1,0,0],*vft.pass2act(*sc)),\
             vft.R([0,1,0],*vft.pass2act(*sc)),\
             vft.R([0,0,1],*vft.pass2act(*sc))]
            
        if m is None:
            for ea,ax,ay,az in zip(eFf,vXF,vYF,vZF):
                for eb,bx,by,bz in zip(eFf,vXF,vYF,vZF):
                    for eag,gz in zip(ea,vZf):
                        for ebd,dz in zip(eb,vZf):
                            out={'eag':eag,'ebd':ebd,'ax':ax,'ay':ay,'az':az,\
                                 'bx':bx,'by':by,'bz':bz,'gz':gz,'dz':dz}
                            yield out
        elif m==-2 or m==2:
            for ea,ax,ay in zip(eFf,vXF,vYF):
                for eb,bx,by in zip(eFf,vXF,vYF):
                    for eag,gz in zip(ea,vZf):
                        for ebd,dz in zip(ea,vZf):
                            out={'eag':eag,'ebd':ebd,'ax':ax,'ay':ay,\
                                 'bx':bx,'by':by,'gz':gz,'dz':dz}
                            yield out
        elif m==-1 or m==1:
            for ea,ax,ay in zip(eFf,vXF,vYF):
                for eb,bz in zip(eFf,vZF):
                    for eag,gz in zip(ea,vZf):
                        for ebd,dz in zip(eb,vZf):
                            out={'eag':eag,'ebd':ebd,'ax':ax,'ay':ay,\
                                 'bz':bz,'gz':gz,'dz':dz}
                            yield out
        elif m==0:      
            for ea,az in zip(eFf,vZF):
                for eb,bz in zip(eFf,vZF):
                    for eag,gz in zip(ea,vZf):
                        for ebd,dz in zip(eb,vZf):
                            out={'eag':eag,'ebd':ebd,'az':az,\
                                 'bz':bz,'gz':gz,'dz':dz}
                            yield out



def D2inf(vZ,nuZ_f=None,nuXZ_f=None,nuZ_F=None,nuXZ_F=None,m=None):
    """
    Estimates the final values of the correlation function:
        <D_{0m}^2(Omega^{v:f,F}_{tau,t+tau})_tau
        
    If nuZ_F is not provided, calculation will be performed in the lab frame 
    (nuXZ_F may also be provided if F defined by two vectors)
    
    If nuZ_f is provide, motion will be calculated for the action of f on the
    bond vector
    
    D2inf=D2inf(vZ,nuZ_f,nuXZ_f=None,nuZ_F=None,nuXZ_F=None,m=None)
    """
    pass

def Ct_D2inf(vZ,vXZ=None,nuZ_f=None,nuXZ_f=None,nuZ_F=None,nuXZ_F=None,mode='both',m=None,mp=0,index=None):
    """
    Calculates the correlation functions and their values at infinite time
    simultaneously (greatly reducing the total number of calculations)
    
    To perform the calculation in reference frame F, provide nuZ_F and 
    optionally nuXZ_F
    
    To calculate the effect of the motion of frame f on the correlation function
    for the bond, provide nuZ_f and optionally nuXZ_f
    
    To only return the correlation function, or only return the values at infinite
    time, set mode to 'Ct' or 'D2inf', respectively.
    
    To only calculate a particular term, set m AND mp (only m required for D2inf).
    
    Setting m OR mp will automatically set the other term to 0. Default is for
    mp=0 (starting component), and m is swept from -2 to 2. 
    
    Currently, m or mp must be zero
    
    index can be provided if the trajectory has been sparsely sampled
    """
    
    if m!=0 and mp!=0:
        print('m or mp must be 0')
        return
    
    if m==0:
        m=mp
        mp=0
        mmpswap=True    #We'll just fix this with sign changes at the end
    else:
        mmpswap=False
    
        
    #Size of the output        
    n=vZ.shape[-1]
    if vZ.ndim==2:
        SZ=[1,2*n]
    else:
        SZ=[vZ.shape[1],2*n] 

    #Flags for what we calculate
    if mode[0].lower()=='b' or mode[0].lower()=='c':
        calc_ct=True
    else:
        calc_ct=False
    
    m0=True if m is None or np.abs(m)==0 else False
    m1=True if m is None or np.abs(m)==1 else False
    m2=True if m is None or np.abs(m)==2 else False

    #Pre-allocation
    if calc_ct:
        ft0=[np.zeros(SZ,dtype=complex) if m0 else None,\
             np.zeros(SZ,dtype=complex) if m1 else None,\
             np.zeros(SZ,dtype=complex) if m2 else None]    #Pre-allocate storage
    d20=[np.zeros(SZ[0],dtype=complex) if m0 else None,\
         np.zeros(SZ[0],dtype=complex) if m1 else None,\
         np.zeros(SZ[0],dtype=complex) if m2 else None]   #D2inf is practically free, so we always calculate

    "l0 is a generator that returns a dictionary with all required components"
    l0=loop_gen(vZ,vXZ,nuZ_f=nuZ_f,nuXZ_f=nuXZ_f,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F,m=m)    
    
    if nuZ_f is None:
        for l in l0:
            zzp=l['az']*l['bz']
            zz=zzp.mean(-1)
            
            if m1:p1=np.sqrt(3/2)*(l['ax']*l['bz']+1j*l['ay']*l['bz'])
            if m2:p2=np.sqrt(3/8)*((l['ax']*l['bx']-l['ay']*l['by'])+1j*2*l['ax']*l['by'])
            
            "Calculate the correlation functions (inverse transform at end)"
            if calc_ct:ftzz=FT(zzp,index=index)
            if m0 and calc_ct:ft0[0]+=3/2*ftzz*ftzz.conj()
            if m1 and calc_ct:ft0[1]+=ftzz*FT(p1,index).conj()
            if m2 and calc_ct:ft0[2]+=ftzz*FT(p2,index).conj()
            
            "Calculate the final values"
            if m0:d20[0]+=3/2*(zz**2)
            if m1:d20[1]+=zz*(p1.mean(-1))
            if m2:d20[2]+=zz*(p2.mean(-1))
    else:
        for l in l0:
            eep=l['eag']*l['ebd']   #This term appears in all calculations
            ee=eep.mean(-1)
            
            gzdz=l['gz']*l['dz']    #Multiplied by all terms below
            
            if m0:p0=3/2*l['az']*l['bz']*gzdz                                                   #0,0 term
            if m1:p1=np.sqrt(3/2)*(l['ax']*l['bz']+1j*l['ay']*l['bz'])*gzdz                     #+1,0 term
            if m2:p2=np.sqrt(3/8)*((l['ax']*l['bx']-l['ay']*l['by'])+1j*2*l['ax']*l['by'])*gzdz   #+2,0 term
                
            "Calculate the correlation functions (inverse transform at end)"
            if calc_ct:ftee=FT(eep,index=index)
            if m0 and calc_ct:ft0[0]+=ftee*FT(p0,index).conj()
            if m1 and calc_ct:ft0[1]+=ftee*FT(p1,index).conj()
            if m2 and calc_ct:ft0[2]+=ftee*FT(p2,index).conj()
            
            "Calculate the final values"
            if m0:d20[0]+=ee*p0.mean(-1)
            if m1:d20[1]+=ee*p1.mean(-1)
            if m2:d20[2]+=ee*p2.mean(-1)
            
    if calc_ct:
        N=get_count(index) if index is not None else np.arange(n,0,-1)
        i=N!=0
        N=N[i]

        ct=[None if ft is None else (np.fft.ifft(ft,axis=-1)[:,:n])[:,i]/N for ft in ft0]
        
    if m0:
        d20[0]+=-1/2
        if calc_ct:ct[0]+=-1/2
    
    "Now correct the signs, fill in other values"
    if mmpswap:     
        d20=[m_mp_swap(d2,0,k,k,0) for k,d2 in enumerate(d20)]
        if calc_ct:ct=[m_mp_swap(ct0,0,k,k,0) for k,ct0 in enumerate(ct)]
        
    if m is None:
        d2=np.concatenate([m_mp_swap(d20[2],0,2,0,-2),m_mp_swap(d20[1],0,1,0,-1),d20[0],d20[1],d20[2]])
        if calc_ct:ct=np.concatenate([m_mp_swap(ct[2],0,2,0,-2),m_mp_swap(ct[1],0,1,0,-1),ct[0],ct[1],ct[2]])
    else:
        d2=m_mp_swap(d20[np.abs(m)],0,np.abs(m),0,m)
        if calc_ct:ct=m_mp_swap(ct[np.abs(m)],0,np.abs(m),0,m)
        
    if vZ.ndim==2:
        d2=d2.squeeze()
        if calc_ct:ct=ct.squeeze()
    
    if mode[0].lower()=='b':
        return ct,d2
    elif mode[0].lower()=='c':
        return ct
    else:
        return d2
    
def m_mp_swap(X,mpi=0,mi=0,mpf=0,mf=0): 
    """
    Performs the appropriate sign changes to switch between components
    of correlation functions or their values at infinite time. One should provide
    the initial component indices (mi,mpi) and the final component indices (mf,mpf)
    
    Currently, one of the components must be 0 or both mi=mpi, mf=mpf
    """
    
    if X is None:
        return None
    
    if not((np.abs(mi)==np.abs(mf) and np.abs(mpi)==np.abs(mpf)) or (np.abs(mi)==np.abs(mpf) and np.abs(mpi)==np.abs(mf))):
        print('Invalid m values')
        print('Example: initial components (0,2) can have final components (0,-2),(2,0),(-2,0)')
        return
    
    if mi!=0 and mpi!=0 and mi!=mpi:
        print('m or mp must be 0, or m=mp')
        return
    
    if mi==mpi and mf==mpf:
        return X
    
    if np.abs(mi)!=np.abs(mf):      #Test for a position swap
#        if np.abs(mi)==1 or np.abs(mf)==1:
#            X=-np.conj(X)   #Sign change and conjugate
#        elif np.abs(mi)==2 or np.abs(mf)==2:
#            X=np.conj(X)    #Conjugate
        X=np.conj(X)
    if (mi+mpi)!=(mf+mpf):  #Test for a sign swap
        if np.abs(mi)==1 or np.abs(mf)==1:
            X=-np.conj(X)   #Sign change and conjugate
        elif np.abs(mi)==2 or np.abs(mf)==2:
            X=np.conj(X)    #Conjugate 
    return X
    
def D2infPAS(vZ,nuZ_F=None,nuXZ_F=None,m=None):
    """
    Estimates the Wigner rotation matrix elements of a bond reorientation between
    two times separated by infinite time, that is, we calculate:
    
    Am=lim_t->infty <D_{0m}^2(\Omega_tau,t+tau>
    
    for a bond with direction given by vZ.
    
    nuZ_F and nuXY_F are optional arguments to define frame in which to calculate
    the terms. Otherwise, calculation is performed in the input frame of vZ. 
    A frame_index must also be provided if nuZ_F is not the same size as vZ
    
    Setting m=None (default) results in all terms being returned in an array 
    (outer dimension runs from m=-2 to 2)
    
    Am = D2inf(vZ,vX=None,vY=None,m=None)
    
    """

    "Apply the frame if required"
    vZ=applyFrame(vZ,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F)
    

    if m is None:
        m1=[-2,-1,0]    #Default, get all values (we'll calculate 1 and 2 from -1 and -2)
    else:
        m1=[m]
        
    if m!=0:
        sc=vft.getFrame(vZ)
        vX=vft.R([1,0,0],*sc)
        vY=vft.R([0,1,0],*sc)
        
    A=list()    #Collect the output


    
    for m0 in m1:   #Sweep over the components (or just one in case m is not None)
        if vZ.ndim==2:
            d2inf=0
        else:
            d2inf=np.zeros(vZ.shape[1],dtype=complex)
        if m0==-2:
            for aX,aY,aZ in zip(vX,vY,vZ):  #Loop over x,y,z components of all 3 axes
                for bX,bY,bZ in zip(vX,vY,vZ):  #ditto
                    d2inf+=np.sqrt(3/8)*((aX*bX).mean(-1)-(aY*bY).mean(-1))*(aZ*bZ).mean(-1)\
                            +1j*np.sqrt(3/2)*(aX*bY).mean(-1)*(aZ*bZ).mean(-1)
        elif m0==-1:
            for aX,aY,aZ in zip(vX,vY,vZ):  #Loop over x,y,z components of all 3 axes
                for bZ in vZ:  #ditto
                    d2inf+=np.sqrt(3/2)*(aX*bZ).mean(-1)*(aZ*bZ).mean(-1)\
                            -1j*np.sqrt(3/2)*(aY*aZ).mean(-1)*(aZ*bZ).mean(-1)
        elif m0==0:
            if vZ.ndim==2:
                d2inf=-0.5   #Offset for 0,0 term
            else:
                d2inf[:]=-0.5
            for aZ in vZ:  #Loop over x,y,z components of all 3 axes
                for bZ in vZ:  #ditto
                    d2inf+=3/2*((aZ*bZ).mean(-1))**2 #Real only
        elif m0==1:
            for aX,aY,aZ in zip(vX,vY,vZ):  #Loop over x,y,z components of all 3 axes
                for bZ in vZ:  #ditto
                    d2inf+=-np.sqrt(3/2)*(aX*bZ).mean(-1)*(aZ*bZ).mean(-1)\
                            -1j*np.sqrt(3/2)*(aY*aZ).mean(-1)*(aZ*bZ).mean(-1)
        elif m0==2:
            for aX,aY,aZ in zip(vX,vY,vZ):  #Loop over x,y,z components of all 3 axes
                for bX,bY,bZ in zip(vX,vY,vZ):  #ditto
                    d2inf+=np.sqrt(3/8)*((aX*bX).mean(-1)-(aY*bY).mean(-1))*(aZ*bZ).mean(-1)\
                            -1j*np.sqrt(3/2)*(aX*bY).mean(-1)*(aZ*bZ).mean(-1)
        A.append(d2inf)
        
    if m is None:
        A.append(-A[1].conj())  #A_1=-A^*_-1
        A.append(A[0].conj())   #A_2=A^*_-2
    else:
        A=A[0]  #Only one element- get rid of list
    
    return np.array(A)  #Return as numpy array
        


def vecs2Ct(vecs):
    """
    Processes the vectors obtained from ini_vec_load to return correlation 
    functions
    """
    pass
    
    
def Ct(vZ,m=0,mp=0,nuZ_F=None,nuXZ_F=None,nuZ_f=None,nuXZ_f=None,index=None,mode='FT'):
    """
    Calculates a correlation function for the bond vector, vZ. This may be done
    under a number of different circumstances. 
    
    First, we may either start with
    component 0 (mp=0) and end with component m, or start in component mp and end
    in component 0 (m=0). Other combinations are not currently implemented.
    
    Second, we may calculate the correlation in a given frame, defined by nuZ_F
    and nuXZ_F. These are optional- otherwise correlation function is evaluated
    in input frame
    
    Third, we may evaluate the effect of the motion of frame f on the bond, vZ
    (in the frame of F or in the input frame). Then, we must input nuZ_f and
    optionally nuXZ_f.
    
    Currently, only evaluation via FT is allowed (mode='FT')
    
    However, we may use sparse sampling of the trajectory, given by index
    
    ct=Ct(vZ,m=0,mp=0,nuZ_F=None,nuXZ_F=None,nuZ_f=None,nuXZ_f=None,frame_index=None,mode='FT')
    
    Output is a numpy array with the desired correlation function (time axis should
    be appended elsewhere). Note, if 3D arrays (dimension x time x bond) are provided, 
    then the output will have dimensions of bondxtime.
    """
    
    if nuZ_f is None:
        return CtPAS(vZ=vZ,m=m,mp=mp,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F,index=index)
    else:
        return Ct_finF(vZ=vZ,nuZ_f=nuZ_f,nuXZ_f=nuXZ_f,m=m,mp=mp,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F,index=index)
    
def Ct_finF(vZ,nuZ_f,nuXZ_f=None,m=0,mp=0,nuZ_F=None,nuXZ_F=None,index=None):
    """
    Calculates the correlation function for the motion of a bond due to motion
    of a frame of a frame f in frame F (or in the lab frame)
    """
    
    "Some checks of the input data"
    if m!=0 and mp!=0:
        print('m or mp must be 0')
        return

    "Apply frame F"
    vZF,nuZ_fF,nuXZ_fF=applyFrame(vZ,nuZ_f,nuXZ_f,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F)
    "Apply frame f"
#    vZf=applyFrame(vZ,nuZ_F=nuZ_f,nuXZ_F=nuXZ_f)
  
    "Axes to project a bond in frame f into frame F"
    sc=vft.getFrame(nuZ_fF,nuXZ_fF)
    vZf=vft.R(vZF,*vft.pass2act(*sc))
    eFf=[vft.R([1,0,0],*vft.pass2act(*sc)),\
         vft.R([0,1,0],*vft.pass2act(*sc)),\
         vft.R([0,0,1],*vft.pass2act(*sc))]
    
    
    "Axes of the bond vector in frame F"
    if not(m==0 and mp==0):
        sc=vft.getFrame(vZF)
        vXF=vft.R([1,0,0],*sc)
        vYF=vft.R([0,1,0],*sc)
        
    n=vZ.shape[-1]
    if vZ.ndim==2:
        SZ=2*n
    else:
        SZ=[vZ.shape[1],2*n] 
    ft0=np.zeros(SZ,dtype=complex)
    if mp==-2 or m==2:
        for ea,ax,ay in zip(eFf,vXF,vYF):
            for eb,bx,by in zip(eFf,vXF,vYF):
                for eag,gz in zip(ea,vZf):
                    for ebd,dz in zip(eb,vZf):
                        ftee=FT(eag*ebd,index)
                        ft0+=np.sqrt(3/8)*FT(ax*bx*gz*dz-ay*by*gz*dz-1j*2*ax*by*gz*dz,index).conj()*ftee
    elif mp==-1 or m==1:
        for ea,ax,ay in zip(eFf,vXF,vYF):
            for eb,bz in zip(eFf,vZF):
                for eag,gz in zip(ea,vZf):
                    for ebd,dz in zip(eb,vZf):
                        ftee=FT(eag*ebd,index)
                        ft0+=np.sqrt(3/2)*FT(ax*bz*gz*dz+1j*ay*bz*gz*dz,index).conj()*ftee
    elif mp==0 and m==0:      
        for ea,az in zip(eFf,vZF):
            for eb,bz in zip(eFf,vZF):
                for eag,gz in zip(ea,vZf):
                    for ebd,dz in zip(eb,vZf):
                        ftee=FT(eag*ebd,index)
                        ft0+=3/2*(FT(az*bz*gz*dz,index).conj()*ftee)        
    elif mp==1 or m==-1:
        for ea,ax,ay in zip(eFf,vXF,vYF):
            for eb,bz in zip(eFf,vZF):
                for eag,gz in zip(ea,vZf):
                    for ebd,dz in zip(eb,vZf):
                        ftee=FT(eag*ebd,index)
                        ft0+=np.sqrt(3/2)*FT(-ax*bz*gz*dz+1j*ay*bz*gz*dz,index).conj()*ftee
    elif mp==2 or m==-2:
        for ea,ax,ay in zip(eFf,vXF,vYF):
            for eb,bx,by in zip(eFf,vXF,vYF):
                for eag,gz in zip(ea,vZf):
                    for ebd,dz in zip(eb,vZf):
                        ftee=FT(eag*ebd,index)
                        ft0+=np.sqrt(3/8)*FT(ax*bx*gz*dz-ay*by*gz*dz+1j*2*ax*by*gz*dz,index).conj()*ftee
    
    "Use to properly normalize correlation function"
    if index is not None:        
        N=get_count(index)
    else:
        N=np.arange(n,0,-1)
    i=N!=0
    N=N[i]

    "Truncate function to half length"
    if vZ.ndim==3:  
        ct=np.fft.ifft(ft0)[:,:n]
        ct=ct[:,i]/N
    else:
        ct=np.fft.ifft(ft0)[:n]
        ct=ct[i]/N
    
    
    "Subtract away 1/2 for m,mp=0"
    if mp==0 and m==0:
        ct=ct.real-0.5
    
    return ct                    

def CtPAS(vZ,m=0,mp=0,nuZ_F=None,nuXZ_F=None,index=None):
    """
    Calculates the correlation function for a bond motion, optionally in frame
    F. Note, this is only for use when we are not interetested in motion of the
    bond in frame F induced by frame F, rather only the bond motion itself, 
    aligned by F. Defintion of F is optional, in which case the resulting 
    correlation function is the lab frame correlation function. 
    
    mp is the starting component, and m is the final component (D^2_{mp,m}). One
    or both must be zero.
    
    Currently, only Ct via FT is implemented. Sparse sampling may be used,
    if specified by index
    """
    
    "Some checks of the input data"
    if m!=0 and mp!=0:
        print('m or mp must be 0')
        return

    vZ=applyFrame(vZ,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F)
        
    if not(m==0 and mp==0):
        sc=vft.getFrame(vZ)
        vX=vft.R([1,0,0],*sc)
        vY=vft.R([0,1,0],*sc)
    
    n=vZ.shape[-1]
    if vZ.ndim==2:
        SZ=2*n
    else:
        SZ=[vZ.shape[1],2*n]
    
    if mp==-2 or m==2:
        ft0=np.zeros(SZ,dtype=complex)
        for aX,aY,aZ in zip(vX,vY,vZ):
            for bX,bY,bZ in zip(vX,vY,vZ):
                ftzz=FT(aZ*bZ,index)
                ft0+=np.sqrt(3/8)*FT(aX*bX-aY*bY+1j*2*aX*bY,index).conj()*ftzz
    elif mp==-1 or m==1:
        ft0=np.zeros(SZ,dtype=complex)
        for aX,aY,aZ in zip(vX,vY,vZ):
            for bZ in vZ:
                ftzz=FT(aZ*bZ,index)
                ft0+=np.sqrt(3/2)*FT(aX*bZ-1j*aY*bZ,index).conj()*ftzz
    elif mp==0 and m==0:
        ft0=np.zeros(SZ)   
        for aZ in vZ:
            for bZ in vZ:
                ftzz=FT(aZ*bZ,index)
                ft0+=3/2*(ftzz.conj()*ftzz).real
    elif mp==1 or m==-1:
        ft0=np.zeros(SZ,dtype=complex)
        for aX,aY,aZ in zip(vX,vY,vZ):
            for bZ in vZ:
                ftzz=FT(aZ*bZ,index)
                ft0+=np.sqrt(3/2)*FT(-aX*bZ-1j*aY*bZ,index).conj()*ftzz
    elif mp==2 or m==-2:
        ft0=np.zeros(SZ,dtype=complex)
        for aX,aY,aZ in zip(vX,vY,vZ):
            for bX,bY,bZ in zip(vX,vY,vZ):
                ftzz=FT(aZ*bZ,index)
                ft0+=np.sqrt(3/8)*FT(aX*bX-aY*bY-1j*2*aX*bY,index).conj()*ftzz
                
    "Truncate function to half length"
    if vZ.ndim==3:  
        ct=np.fft.ifft(ft0)[:,:n]
    else:
        ct=np.fft.ifft(ft0)[:n]
    
    "Properly normalize correlation function"
    if index is not None:        
        N=get_count(index)
    else:
        N=np.arange(n,0,-1)
    ct=ct/N    
    
    "Subtract away 1/2 for m,mp=0"
    if mp==0 and m==0:
        ct=ct.real-0.5
    
    return ct

def FT(x,index=None):
    """
    Performs a zero-filled Fourier transform (doubling the size). If an index
    is included, a matrix is created with zeros at all positions not
    in index, and having the values of the input vector, x, elsewhere.
    
    X = FT(x,index=None)
    """
    if index is not None:
        if x.ndim==1:
            x1=np.zeros(index.max()+1)
            x1[index]=x
        else:
            x1=np.zeros([x.shape[0],index.max()+1])
            x1[:,index]=x
    else:
        x1=x
    n=x1.shape[-1]
    return np.fft.fft(x1,2*n,axis=-1)

def Ct_similar(ct0,A,m=None):
    """
    Calculate correlation functions from Ct_{00} assuming that all functions
    have a similar shape (Ct_00 starts from one and decays to A[2]), other 
    correlation functions start at 0 and increase to A[m]. Required arguments
    are ct0 (=Ct_{00}), and the 5 elements of A. Optionally, may provide
    multiple correlation functions with multiple A. All 5 correlation functions
    are returned unless m is specified
    
    Ct=Ct_similar(ct0,A,m=None)
    """
    
    if m==0:
        return ct0  #Not sure why you'd call this function, but then return original
    
    A=np.array(A)
    if A.ndim==2:
        if A.shape[1]==5:
            A=A.T   #Last dimension over bonds
        if ct0.shape[0]==A.shape[1]:
            ct0=ct0.T #Last dimension over bonds (for broadcasting)
            tr=True
        else:
            tr=False
        
    
    ct_norm=np.array(((ct0-A[2])/(1-A[2])).real,dtype=complex)
    
    if m is None:
        ct=np.array([(1-ct_norm)*a for a in A])
        ct[2]=ct0
    else:
        ct=(1-ct_norm)*A[m+2]   #Just one element, add 2 to get correct index
        
    if tr:ct=ct.swapaxes(-2,-1)
    
    return ct

def applyFrame(*vecs,nuZ_F=None,nuXZ_F=None):
    """
    Applies a frame, F, to a set of vectors, *vecs, by rotating such that the
    vector nuZ_F lies along the z-axis, and nuXZ_F lies in the xz-plane. Input
    is the vectors (as *vecs, so list separately, don't collect in a list), and
    the frame, defined by nuZ_F (a vector on the z-axis of the frame), and 
    optionally nuXZ_F (a vector on xy-axis of the frame). These must be given
    as keyword arguments.
    
    vecs_F = applyFrame(*vecs,nuZ_F=nuZ_F,nuXZ_F=None,frame_index=None)
    
    Note, one may also omit the frame application and just apply a frame index
    """
    if nuZ_F is None:
        out=vecs
    else:
        sc=vft.pass2act(*vft.getFrame(nuZ_F,nuXZ_F))
        out=[None if v is None else vft.R(v,*sc) for v in vecs]
        
    if len(vecs)==1:
        return out[0]
    else:
        return out

def ini_vec_load(traj,frame_funs,frame_index=None,index=None,dt=None):
    """
    Loads vectors corresponding to each frame, defined in a list of frame functions.
    Each element of frame_funs should be a function, which returns one or two
    vectors.
    
    traj should be the trajectory from MDanalysis

    frame_index is a list of indices, allowing one to match each bond to the correct
    frame (length of each list matches number of bonds, values in list should be between
    zero and the number of frames-1)

    index (optional) is used for sparse sampling of the trajectory
    
    dt gives the step size of the MD trajectory (use to override dt found in traj)
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
