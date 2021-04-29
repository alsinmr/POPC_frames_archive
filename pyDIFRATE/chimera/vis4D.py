#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:58:00 2020

@author: albertsmith
"""

import os
import cv2
import MDAnalysis as mda
import matplotlib.pyplot as plt
curdir=os.getcwd()
import numpy as np
os.chdir('../Struct')
from vf_tools import Spher2pars,norm,getFrame,Rspher,pbc_corr
import vf_tools as vft
import eval_fr as ef
import select_tools as selt
os.chdir('../chimera')
from chimeraX_funs import py_line,WrCC,chimera_path,run_command,copy_funs,write_tensor
from chimeraX_funs import sel_indices,py_print_npa,guess_disp_mode
from shutil import copyfile
os.chdir(curdir)

def time_axis(nt0=1e5,nt=300,step='log',dt=0.005,fr=15,mode='time'):
    """
    Constructs a linear- or log-spaced time axis from an initial time axis. One
    may return the time axis itself, or an index for accessing the appropriate
    frames out of the original time axis.
    
        nt0     :   Number of (used) time points in original trajectory.
        nt      :   Number of time points in desired axis
        step    :   Type of step (linear or log)
        dt      :   Step size (in ns, 'time' and 'avg' only)
        fr      :   Frame rate (frames/s, '
        mode    :   Type of axis. 'time' will simply return the time axis itself
                    'index' will return an index for accessing the desired time
                    points from the trajectory. 'avg' will return the time elapsed
                    in the trajectory per second. 'avg_index' will return the
                    index corresponding specifically for 'avg'
                    
    """
    
    if mode=='index' or mode=='avg_index':dt=1 #If the mode is just to return the index, then we don't consider the real time step
    
    
    if step.lower()=='log':
        t=(np.logspace(0,np.log10(nt0),nt,endpoint=True)-1)*dt
    else:
        t=(np.linspace(0,nt0,nt,endpoint=False))*dt
    
    if mode.lower()=='time':return t
    
    if mode.lower()=='index':
        t=np.round(t).astype(int)
#        if step=='log':t+=-1
        return t

    if mode.lower()=='avg' or 'avg_index':
        Dt=list()
        for k in range(len(t)):
            i1=np.max([0,k-int(fr/2)])
            i2=np.min([k+int(fr/2),len(t)-1])
            Dt.append(fr*(t[i2]-t[i1])/(i2-i1))
            
        Dt=np.array(Dt)
        
        if mode.lower()=='avg':return Dt
        t0=np.arange(nt0)*dt
        t=np.array([np.argmin(np.abs(Dt0-t0)).squeeze() for Dt0 in Dt])
        return t
    else:
        print('Unrecognized mode: used "time", "index", "avg", or "avg_index"')
        return    
        

def md2images(mol,sel0,ct_m0=None,index=None,nt0=1e5,nt=300,step='log',sc=2.09,\
                file_template='images/tensor_{0:06d}.jpg',scene=None,\
                save_opts='width 1000 height 600 supersample 2',chimera_cmds=None,\
                marker=None,pdb_template=None,reorientT=True):
        
    sel=sel0

    "Setup the residue index"
    if ct_m0 is not None:
        ct_m0=np.array(ct_m0)
        nr=ct_m0.shape[1]
        if index is None:
            index=np.arange(nr,dtype=int)
        else:
            if len(index)==nr and np.max(index)<2:
                index=np.array(index,dtype=bool)
            else:
                index=np.array(index,dtype=int)
           
        ct_m0=ct_m0[:,index]
    
    "Setup time axis"
    t_index=time_axis(nt0=nt0,nt=nt,step=step,mode='index')
        
    file_template=os.path.realpath(file_template)
    folder,_=os.path.split(file_template)
    file_template="'"+file_template+"'"
    
    
    "Get indices to connect sel to sel1 and to sel2"
    if ct_m0 is not None:
        s1i=np.array([np.argwhere(s1.index==sel.indices).squeeze() for s1 in mol.sel1])
        s2i=np.array([np.argwhere(s2.index==sel.indices).squeeze() for s2 in mol.sel2])
        if pdb_template is not None:
            sel3=selt.find_bonded(mol.sel2,sel0,exclude=mol.sel1,n=1,sort='cchain')[0]
            s3i=np.array([np.argwhere(s3.index==sel.indices).squeeze() for s3 in sel3])
            
    if not(os.path.exists(folder)):os.mkdir(folder) #Make sure directory exists
    
    if ct_m0 is not None or pdb_template is None:
#    if not(ct_m0 is None and pdb_template is not None):
        mp0=np.array([0,0,0])
        "Start the loop over pdbs"
        for k,t0 in enumerate(t_index): 
            if pdb_template is None:
                mol.mda_object.trajectory[t0]
        

                
                "Try to keep molecule in center of box, correct boundary condition errors"
                if k!=0:
                    sel.positions=pbc_corr(sel.positions-mp0,mol.mda_object.dimensions[:3])
                    mp=sel.positions.mean(0)
                    mp0=mp+mp0
                    sel.positions=sel.positions-mp
                else:
                    mp0=sel.positions.mean(0)
                    sel.positions=pbc_corr(sel.positions-sel.positions.mean(0),\
                                           mol.mda_object.dimensions[:3])
                    
            else:
                sel=mda.Universe(pdb_template.format(k)).atoms
                
            
            if ct_m0 is not None:
                
                "Get the current positions of the middle of each bond"
                pos=(sel[s1i].positions+sel[s2i].positions).T/2
        
                
                "Get the current orientations of the bonds, rotate tensors accordingly"
                if pdb_template is None:
                    vZ,vXZ=mol._vft()
                else:
                    if reorientT:
                        vZ=(sel[s2i].positions-sel[s1i].positions).T
                        vXZ=(sel[s3i].positions-sel[s1i].positions).T
                    elif k==0:
                        mol.mda_object.trajectory[0]
                        vZ,vXZ=mol._vft()
                        vZ0=norm(vZ.copy())
                        vXZ0=vXZ.copy()
                    else:
                        vZ=norm((sel[s2i].positions-sel[s1i].positions).T)
                        cb=(vZ0*vZ).sum(0)
                        sb=np.sqrt(1-cb**2)
                        vec=np.array([vZ0[m]*vZ[n]-vZ0[n]*vZ[m] for m,n in zip([1,2,0],[2,0,1])]) #Cross product
                        sc0=vft.getFrame(vec)  #Rotate around a vector perpendicular to starting and current position
                        vXZ=vft.R(vft.Rz(vft.R(vXZ0,*vft.pass2act(*sc0)),cb,sb),*sc0) #Frame of vec, cb/sb, frame of vec
                        
                vZ=norm(vZ)
                scF=getFrame(vZ[:,index],vXZ[:,index])
                tensors=Rspher(ct_m0[:,:,t0],*scF)
                
                "Convert tensors into parameters"
                delta,eta,*euler=Spher2pars(tensors,return_angles=True)
                
                "Write tensors to file"
                write_tensor(os.path.join(folder,'tensors_{0:06d}.txt'.format(k)),delta*sc,eta,euler,pos,marker)
        
            if pdb_template is None:
                sel.write(os.path.join(folder,'pdb{0:06d}.pdb'.format(k))) 

    
    if pdb_template is None:
        pdb_template=os.path.join(folder,'pdb{0:06d}.pdb')
        rmpdb=True
    else:
        if not(os.path.isabs(pdb_template)):
            pdb_template=os.path.join(folder,pdb_template)
        rmpdb=False
    pdb_template="'"+pdb_template+"'"
    if scene is not None:
        scene="'"+scene+"'"

    "Now write the chimera script"
    full_path=os.path.join(folder,'chimera_script.py')    
    
    with open(full_path,'w') as f:
        py_line(f,'import os')
        py_line(f,'import numpy as np')
        py_line(f,run_command(version='X'))
        
        copy_funs(f)    #Copy required functions into chimeraX script
                
        
        py_line(f,'\ntry:')
#        py_line(f,'\nif True:')

        
        py_line(f,'for k in range({0:d}):'.format(nt),1)
        if scene is not None:
#            py_line(f,'session.open_command.open_data("'+scene+'")',2)
            WrCC(f,'open '+scene,2)
        
        
        f.write('\t\trc(session,"open {0}".format(k))\n'.format(pdb_template))
#        f.write('\t\tsession.open_command.open_data("{0}".format(k))\n'.format(pdb_template))


        if chimera_cmds is not None:
            if isinstance(chimera_cmds,str):chimera_cmds=[chimera_cmds]
            for cmd in chimera_cmds:
                WrCC(f,cmd,2)
        
        WrCC(f,'display',2)
        
        

        
        positive_color=(255,100,100,255)
        negative_color=(100,100,255,255)
        if ct_m0 is not None:
            py_line(f,'load_surface(session,"{0}".format(k),sc={1},theta_steps={2},phi_steps={3},positive_color={4},negative_color={5})'\
                        .format(os.path.join(folder,'tensors_{0:06d}.txt'),sc,50,25,positive_color,negative_color),2)
        
        


        f.write('\t\trc(session,"save '+"{0} ".format(file_template)+'{0}".format(k))\n'.format(save_opts))
        
#        f.write('\t\tsession.save_command.save_data("{0}".format(k))\n'.format(file_template))
        WrCC(f,'close',2)
            
        py_line(f,'except:')
        py_line(f,'print("Error in chimera script")',1)
        py_line(f,'finally:')
        if ct_m0 is not None or rmpdb:
            py_line(f,'for k in range({0}):'.format(nt),1)
        if ct_m0 is not None:
            py_line(f,'os.remove("{0}".format(k))'.format(os.path.join(folder,'tensors_{0:06d}.txt')),2)
        if rmpdb:
            py_line(f,'os.remove("{0}".format(k))'.format(os.path.join(folder,'pdb{0:06d}.pdb')),2)
        WrCC(f,'exit',1)
                

        
        
    "Copy the created chimera files to names in the chimera folder (ex. for debugging)"
    os.spawnl(os.P_NOWAIT,chimera_path(version='X'),chimera_path(version='X'),full_path)


def det_fader(data,mol,fr=15,nt0=1e5,nt=300,step='log',scaling=1,\
                file_template='images/det_fade{0:06d}.jpg',scene=None,\
                save_opts='width 1000 height 600 supersample 2',chimera_cmds=None,\
                marker=None,pdb_template=None,disp_mode=None):
    """
    Takes a molecule object or a set of pdb files (given as a template string) and 
    generates images where, as we move through the time axis, we fade between
    3D plots of the individual detector responses. We need the data object
    to achieve this, in addition to the molecule object and a MDAnalysis selection
    for writing 
    """
    
    
    "Here we try to guess the display mode if not given"
    if disp_mode is None:
        disp_mode=guess_disp_mode(mol)
    
    di=sel_indices(mol,disp_mode,mode='all')
    
    "Setup time axis"
    t_index=time_axis(nt0=nt0,nt=nt,step=step,mode='index')
    t=time_axis(fr=fr,nt0=nt0,nt=nt,step=step,mode='avg')
        
    file_template=os.path.realpath(file_template)
    folder,_=os.path.split(file_template)
    file_template="'"+file_template+"'"
    
    if not(os.path.exists(folder)):os.mkdir(folder) #Make sure directory exists
    
    "Get colors for plotting"
    
    clr0=list()
    clr=plt.rcParams['axes.prop_cycle'].by_key()['color']
    for k in range(data.R.shape[1]):
        clr0.append(np.append(hex_to_rgb(clr[k]),255))
    clr0=np.array(clr0)    
    
    clr=clr0.T
    "Determine indices of the atoms to be colored"
    id0=sel_indices(mol,disp_mode,mode='value')
#    x=np.concatenate([x0*np.ones(len(i)) for i,x0 in zip(id0,x)])
    id1=np.concatenate([i for i in id0]).astype(int)
    ids,b=np.unique(id1,return_index=True)
#    x=np.array([x[id0[b0]==id0].mean() for b0 in b])
    
    "Calculate colors and radii as a function of time"
    rhoz,_=data.sens._rho_eff(mdl_num=None)
    tc=data.sens.tc()
#    clr1=np.array([(c*rhoz.T).sum(1)/rhoz.sum(axis=0) for c in clr0.T]).T
    clr0=[210,180,140,255]  #Tan for less-than-1 total response
    
#    x=np.zeros([t.size,len(ids)])
#    clr=np.zeros([t.size,4],dtype='uint8')
    R=list()
    for R0 in data.R.copy().T:
        R0=np.concatenate([R1*np.ones(len(i)) for i,R1 in zip(id0,R0)])
        R0=np.array([R0[id1[b0]==id1].mean() for b0 in b])
        R.append(R0*scaling)
    R=np.array(R).T
#    R0*=scaling
#    for k,t0 in enumerate(t):
#        b1=np.argmin(np.abs(t0/1e9-data.sens.tc()))
#        R=(R0*rhoz[:,b1]).sum(1)
#        R=np.concatenate([R0*np.ones(len(i)) for i,R0 in zip(id0,R)])
#        x[k]=np.array([R[id1[b0]==id1].mean() for b0 in b])
#        clr[k]=clr1[b1]
    
    
    "Write out pdbs for plotting if required"    
    if pdb_template is None:
        pdb_template=os.path.join(folder,'pdb{0:06d}.pdb')
        rmpdb=True
        sel=mol.mda_object.atoms[di]
        for k,t0 in enumerate(t_index):
            mol.mda_object.trajectory[t0]
            
            
            "Try to keep molecule in center of box, correct boundary condition errors"
            if k!=0:
                sel.positions=pbc_corr(sel.positions-mp0,mol.mda_object.dimensions[:3])
                mp=sel.positions.mean(0)
                mp0=mp+mp0
                sel.positions=sel.positions-mp
            else:
                mp0=sel.positions.mean(0)
                sel.positions=pbc_corr(sel.positions-sel.positions.mean(0),\
                                       mol.mda_object.dimensions[:3])
                
            sel.write(pdb_template.format(k))
    else:
        rmpdb=False
        
    "Using quotes allows us to have spaces in paths below"
    pdb_template="'"+pdb_template+"'"
    if scene is not None:
        scene="'"+scene+"'"    
    
    "Now write the chimera script"
    full_path=os.path.join(folder,'chimera_script{0:06d}.py'.format(np.random.randint(1e6)))
    
    with open(full_path,'w') as f:
        py_line(f,'import os')
        py_line(f,'import numpy as np')
        py_line(f,run_command(version='X'))
        
#        py_print_npa(f,'x',x,format_str='.6f',dtype='float',nt=0)
        py_print_npa(f,'tc',tc,format_str='.6e',dtype='float',nt=0)
        py_print_npa(f,'rhoz',rhoz,format_str='.6f',dtype='float',nt=0)
        py_print_npa(f,'t',t,format_str='.6e',dtype='float',nt=0)
        py_print_npa(f,'R',R,format_str='.6f',dtype='float',nt=0)
        py_print_npa(f,'clr',clr,format_str='d',dtype='uint8',nt=0)
        py_print_npa(f,'clr0',clr0,format_str='d',dtype='uint8',nt=0)
        py_print_npa(f,'di',di,format_str='d',dtype='uint32',nt=0)
        py_print_npa(f,'ids',ids,format_str='d',dtype='uint32',nt=0)
        
        py_line(f,'if True:')
#        py_line(f,'try:')
        
        "Start the loop in chimeraX over time"
#        py_line(f,'for k,(c,x0) in enumerate(zip(clr,x)):',1)
        py_line(f,'for k,t0 in enumerate(t):',1)
        if scene is not None:
#            py_line(f,'session.open_command.open_data("'+scene+'")',2)
            WrCC(f,'open '+scene,2)
        
        "Open the pdb here"
        f.write('\t\trc(session,"open {0}".format(k))\n'.format(pdb_template))
        
        "Write chimera commands"
        if chimera_cmds is not None:
            if isinstance(chimera_cmds,str):chimera_cmds=[chimera_cmds]
            for cmd in chimera_cmds:
                WrCC(f,cmd,2)
        
#        WrCC(f,'~display',2)
        WrCC(f,'~ribbon',2)
        
        "Get the atoms to be displayed"
        py_line(f,'while len(session.models)>1:',2)
        py_line(f,'session.models[0].delete()',3)
        py_line(f,'atoms=session.models[0].atoms',2)
        WrCC(f,'display #1',2)
             
        py_line(f,'hide=getattr(atoms,"hides")',2)
        py_line(f,'hide[:]=1',2)
        py_line(f,'hide[di]=0',2)
        py_line(f,'setattr(atoms,"hides",hide)',2)
        
        #Parameter encoding
        WrCC(f,'style ball',2)
        WrCC(f,'size stickRadius 0.2',2)
        WrCC(f,'color all tan',2)

        #Calculate radius and colors here
        
        py_line(f,'b=np.argmin(np.abs(t0/1e9-tc))',2)
        py_line(f,'x0=(rhoz[:,b]*R)',2)
        py_line(f,'r=4*x0.sum(1)+0.9',2)
        py_line(f,'c1=np.array([c0*(1-x0.sum(1))+(c01*x0).sum(1) for c0,c01 in zip(clr0,clr)]).T.astype("uint8")',2)
    #        py_line(f,'x0[x0<0]=0',2)
#        py_line(f,'x0[x0>1]=1',2)
#        py_line(f,'c1=np.array([c00*(1-x0)+c10*x0 for c00,c10 in zip(clr0,c)]).T',2)
#        py_line(f,'r=4*x0+0.9',2)
        py_line(f,'r0=getattr(atoms,"radii").copy()',2)
        py_line(f,'r0[:]=.8',2)
        py_line(f,'r0[ids]=r',2)
        py_line(f,'c0=getattr(atoms,"colors").copy()',2)
        py_line(f,'c0[:]=clr0',2)
        py_line(f,'c0[ids]=c1',2)
        py_line(f,'setattr(atoms,"radii",r0)',2)
        py_line(f,'setattr(atoms,"colors",c0)',2)
                
        f.write('\t\trc(session,"save '+"{0} ".format(file_template)+'{0}".format(k))\n'.format(save_opts))
        
        if rmpdb:
            py_line(f,'for k in range({0}):'.format(nt),1)
            py_line(f,'os.remove("{0}".format(k))'.format(pdb_template),2)
        py_line(f,'os.remove("{0}")'.format(full_path),1)
        WrCC(f,'exit',1)
        
        copyfile(full_path,full_path[:-9]+'.py')
                

        
        
    "Copy the created chimera files to names in the chimera folder (ex. for debugging)"
    os.spawnl(os.P_NOWAIT,chimera_path(version='X'),chimera_path(version='X'),full_path)
        


def images2movie(file_template,fileout,fr=15,nt=None):
    """
    Takes a series of figures (numbered 0 to N, unless otherwise specified),
    and creates a movie from the images. Currently writes to avi files.
    """
    
    file_template=os.path.realpath(file_template)
    folder,_=os.path.split(file_template)
    
    if not(len(fileout)>3 and fileout[-4:]=='.mp4'):
        fileout+='.mp4'
    
    im0=cv2.imread(file_template.format(0))
    height, width, layers = im0.shape
    size = (width,height)
    
    
    out = cv2.VideoWriter(fileout,cv2.VideoWriter_fourcc(*'MP4V'), fr, size)
    try:
        i=0
        while os.path.exists(file_template.format(i)) and (nt is None or i<nt):
            im=cv2.imread(file_template.format(i))
            out.write(im)
            i+=1
    except:
        print('Video writing failed')
    finally:
        out.release()

def multi_images(files,fileout,sc=1,locations='ne',alpha=1,SZ0=None,clear=True):
    """
    Takes two image files and places one image on top of the other image (file2
    sits on top of file1. One may simply replace the pixels in file1, or one
    may specify alpha, so that file2 and file1 are averaged). One must specify
    the two files, the fileout, sc, which scales the input size of file2, and
    location ('n','s','e','w','nw','ne','se', etc., 'm' for middle, OR a list 
    of two numbers specifying the position of the middle of the image, values 
    between 0 and 1). Finally, one should specify alpha to determine averageing 
    of the two images (alpha=1 replaces image1 entirely with image2, alpha=0 
    will only show image1)
    
    combine_image(file1,file2,fileout,sc=1,location='nw',alpha=1)
    """
    
    IM=[cv2.imread(f) for f in files]
    
    SZ=[im.shape[:-1] for im in IM]
    
    if not(hasattr(sc,'__len__')):
        sc=[sc for _ in range(len(SZ))]
    
    SZ=[((np.array(sz[1])*sc0).astype(int),(np.array(sz[0])*sc0).astype(int))\
         for sc0,sz in zip(sc,SZ)]
    IM=[cv2.resize(im,sz) for im,sz in zip(IM,SZ)]
    SZ=[[sz[1],sz[0]] for sz in SZ]
    
    if SZ0 is None:
        SZ0=SZ.pop(0)
        imout=IM.pop(0)
    else:
        imout=255*np.ones(np.concatenate((SZ0,[3])),dtype='uint8')
        

    
    for location,im,sz in zip(locations,IM,SZ):
        if isinstance(location,str):
            if location.lower()=='n':
               xoff=int((SZ0[1]-sz[1])/2)
               yoff=0
            elif location.lower()=='s':
               xoff=int((SZ0[1]-sz[1])/2)
               yoff=SZ0[0]-sz[0]
            elif location.lower()=='w':
               xoff=0
               yoff=int((SZ0[0]-sz[0])/2)
            elif location.lower()=='e':
               xoff=SZ0[1]-sz[1]
               yoff=int((SZ0[0]-sz[0])/2)
            elif location.lower()=='nw':
               xoff=0
               yoff=0
            elif location.lower()=='ne':
               xoff=SZ0[1]-sz[1]
               yoff=0
            elif location.lower()=='sw':
               xoff=0
               yoff=SZ0[0]-sz[0]
            elif location.lower()=='se':
               xoff=SZ0[1]-sz[1]
               yoff=SZ0[0]-sz[0]
            elif location.lower()=='m':
               xoff=int((SZ0[1]-sz[1])/2)
               yoff=int((SZ0[0]-sz[0])/2)
            else:
               print('Location not recognized')
               return
        else:
            xoff=int(location[0]*(SZ0[1]-sz[1])+sz[1]/2)
            yoff=int(location[1]*(SZ0[0]-sz[0])+sz[0]/2)
        x1b,x2b=(0,-xoff) if xoff<0 else (xoff,0)
        x1e,x2e=(SZ0[1],SZ0[1]-xoff) if xoff+sz[1]>SZ0[1] else (xoff+sz[1],sz[1])
        y1b,y2b=(0,-yoff) if yoff<0 else (yoff,0)
        y1e,y2e=(SZ0[0],SZ0[0]-yoff) if yoff+sz[0]>SZ0[0] else (yoff+sz[0],sz[0])
        
        
        if clear:
            imout0=imout[y1b:y1e,x1b:x1e].copy()
            
        imout[y1b:y1e,x1b:x1e]=((1-alpha)*imout[y1b:y1e,x1b:x1e]).astype(int)
    
        imout[y1b:y1e,x1b:x1e]=imout[y1b:y1e,x1b:x1e]+(alpha*im[y2b:y2e,x2b:x2e]).astype(int)
    
        if clear:
            ci=np.all(im[y2b:y2e,x2b:x2e]>200,2)    
            (imout[y1b:y1e,x1b:x1e])[ci]=imout0[ci]
            
    cv2.imwrite(fileout,imout)
        
def combine_image(file1,file2,fileout,sc=1,location='ne',alpha=1,clear=True):
    """
    Takes two image files and places one image on top of the other image (file2
    sits on top of file1. One may simply replace the pixels in file1, or one
    may specify alpha, so that file2 and file1 are averaged). One must specify
    the two files, the fileout, sc, which scales the input size of file2, and
    location ('n','s','e','w','nw','ne','se', etc., 'm' for middle, OR a list 
    of two numbers specifying the position of the middle of the image, values 
    between 0 and 1). Finally, one should specify alpha to determine averageing 
    of the two images (alpha=1 replaces image1 entirely with image2, alpha=0 
    will only show image1)
    
    combine_image(file1,file2,fileout,sc=1,location='nw',alpha=1)
    """
    
    im1=cv2.imread(file1)
    im2=cv2.imread(file2)
    
    *SZ1,nc1=im1.shape
    *SZ2,nc2=im2.shape
    
    SZ2=((np.array(SZ2[1])*sc).astype(int),(np.array(SZ2[0])*sc).astype(int))
    im2=cv2.resize(im2,SZ2)
    SZ2=[SZ2[1],SZ2[0]]

    "Determine offsets of two images"
    if isinstance(location,str):
        if location.lower()=='n':
           xoff=int((SZ1[1]-SZ2[1])/2)
           yoff=0
        elif location.lower()=='s':
           xoff=int((SZ1[1]-SZ2[1])/2)
           yoff=SZ1[0]-SZ2[0]
        elif location.lower()=='w':
           xoff=0
           yoff=int((SZ1[0]-SZ2[0])/2)
        elif location.lower()=='e':
           xoff=SZ1[1]-SZ2[1]
           yoff=int((SZ1[0]-SZ2[0])/2)
        elif location.lower()=='nw':
           xoff=0
           yoff=0
        elif location.lower()=='ne':
           xoff=SZ1[1]-SZ2[1]
           yoff=0
        elif location.lower()=='sw':
           xoff=0
           yoff=SZ1[0]-SZ2[0]
        elif location.lower()=='se':
           xoff=SZ1[1]-SZ2[1]
           yoff=SZ1[0]-SZ2[0]
        elif location.lower()=='m':
           xoff=int((SZ1[1]-SZ2[1])/2)
           yoff=int((SZ1[0]-SZ2[0])/2)
        else:
           print('Location not recognized')
           return
    else:
        xoff=int(location[0]*(SZ1[1]-SZ2[1])+SZ2[1]/2)
        yoff=int(location[1]*(SZ1[0]-SZ2[0])+SZ2[0]/2)
        
        

    x1b,x2b=(0,-xoff) if xoff<0 else (xoff,0)
    x1e,x2e=(SZ1[1],SZ1[1]-xoff) if xoff+SZ2[1]>SZ1[1] else (xoff+SZ2[1],SZ2[1])
    y1b,y2b=(0,-yoff) if yoff<0 else (yoff,0)
    y1e,y2e=(SZ1[0],SZ1[0]-yoff) if yoff+SZ2[0]>SZ1[0] else (yoff+SZ2[0],SZ2[0])

    im=im1.copy()
    im[y1b:y1e,x1b:x1e]=((1-alpha)*im[y1b:y1e,x1b:x1e]).astype(int)
    
    im[y1b:y1e,x1b:x1e]=im[y1b:y1e,x1b:x1e]+(alpha*im2[y2b:y2e,x2b:x2e]).astype(int)
    
    if clear:
        ci=np.all(im2[y2b:y2e,x2b:x2e]>200,2)    
        (im[y1b:y1e,x1b:x1e])[ci]=(im1[y1b:y1e,x1b:x1e])[ci]
    
    cv2.imwrite(fileout,im)
    
def tile_image(filenames,fileout,grid=None,pos=None,sc=None,SZ=None,clear=True,
               tot_sc=1):
    """
    Takes a list of images (filenames), and either tiles those images (stretching
    so that each has the same shape as the first), or one may explicitly specify
    the position of each image, relative to the absolute size (SZ). 
    
    Tile mode: If pos is set to None, then we use tile mode. In this case, the
    images are placed on a grid. If no further arguments are given, the first
    image will not be scaled, and subsequent images will be forced to fit into
    the same space as the first image. Note we may use tiling while directly 
    defining some positions. In this case, use a list of positions, but those
    positions that should be tiled should be set to None, and those not tiled
    should be given explicitely. Note: an image's position in the filename list
    determines its grid position. If we have a 3x2 grid, and the 4th element of 
    the list has a position explicitely given, then that position in the grid
    will be omitted. Also, if the grid size is not given explicitly, then it
    will be made big enough to fit all images without positions specified.
    
    Position mode: Specify the size of the final image (500x500 default). Then,
    provide a list of positions for each image. Each position has 
    elements (horizontal and vertical position of upper right-hand corner of 
    image, given between 0 and 1).
    
    grid:   Specify the shape of the grid (two elements, whose product should
            be greater than or equal to the number of files, N). Will be 
            automatically set in tile mode.
    
    sc:     May be one element (scales all images by a fixed factor), two elements 
            (scales all images by a separate height and width), or a list of
            N elements, with one or two scaling factors each. Setting to 1 in
            grid mode will prevent the images from scaling (but may cut off images)
            
    pos:    Specify the position of each image. Set to None if image is to be
            tiled
            
    SZ:     Final size of the output image. If set to None, then this will be
            set so that the first image in the grid is unscaled (if no grid used,
            this is just the size)
            
    clear:  Set white space to be see-through, such that one can see images below
            othe images. Note, the first image is on the bottom, and last image
            on top
    tot_sc: Overall scaling (applied at last step). 
    """
    
    N=len(filenames)    #Number of images

    if pos is None:pos=[None for _ in range(N)] #Set later to values
    
    ng=0
    for k in range(N):ng=k if pos[k] is None else ng    #Find last element of pos that's None
    
    if grid is None:    #Set the grid size
        grid=[np.ceil(np.sqrt(ng)),np.floor(np.sqrt(ng))]
        if np.prod(grid)<ng:grid[1]+=1
        if np.prod(grid)==0:grid=[1,1]
    else:
        grid=[grid[1],grid[0]]
    
    fileout=os.path.realpath(fileout)
    folder,_=os.path.split(fileout)
    if not(os.path.exists(folder)):os.mkdir(folder)
    
    im=[cv2.imread(f) for f in filenames]
    SZ0=[im0.shape[1::-1] for im0 in im]
    "Make sure all images specify alpha"
    im=[np.concatenate((im0,255*np.ones([sz0[1],sz0[0],1],dtype='uint8')),axis=-1) if im0.shape[-1]==3 else im0 for im0,sz0 in zip(im,SZ0)]  
    

    
    
    
    if sc is None:sc=[None for _ in range(N)]   #Default to scale-to-fit
    if len(sc)==2 and N!=2:sc=[sc for _ in range(N)]    #If all scaling factors are the same
    sc=[[sc0,sc0] if (sc0 is not None and not(hasattr(sc0,'__len__'))) else None for sc0 in sc]
        
    "Scale the images"
    SZ00=SZ0[0]
    for k,(im0,sc0,pos0,sz0) in enumerate(zip(im,sc,pos,SZ0)):
        if sc0 is None and pos0 is None:    #Image in grid without scaling specified
            sz0=np.round(np.array(SZ00)).astype(int) 
        elif sc0 is not None:
            sz0=np.round(np.array(sz0)*np.array(sc0)).astype(int) #Calculate new size
        else:
            break
        SZ0[k]=sz0
        im[k]=cv2.resize(im0,(sz0[0],sz0[1]),interpolation=cv2.INTER_AREA) #Apply new size
    
    if SZ is None:SZ=np.array([SZ00[0]*grid[0],SZ00[1]*grid[1]],dtype=int)        
    "Set positions for images to be tiled"
    pos=[None if p is None else [p[1],p[0]] for p in pos]   #Swap order
    pos=[[np.floor(k/grid[1])/grid[0],np.mod(k,grid[1])/grid[1]] if p is None\
          else p for k,p in enumerate(pos)]
#    pos=[[np.mod(k,grid[0])/grid[0],np.floor(k/grid[0])/grid[1]] if p is None else p for k,p in enumerate(pos)]
    
    imout=255*np.ones(np.concatenate((SZ[::-1],[4])),dtype='uint8') #Imagout with white background
    
    
    "Add together all images"
    for k in range(N):
        s0,s1=np.round(pos[k]*SZ).astype(int)
        e0,e1=np.min([s0+SZ0[k][0],SZ[0]]),np.min([s1+SZ0[k][1],SZ[1]])
        if clear:
            ci=np.all(im[k][:e1-s1,:e0-s0]>200,2)    #White space
            (im[k][:e1-s1,:e0-s0])[ci]=(imout[s1:e1,s0:e0])[ci]    #Replace with the final image
        imout[s1:e1,s0:e0]=im[k][:e1-s1,:e0-s0] #Write into final image
    
    

    SZ=tuple([int(sz*tot_sc) for sz in SZ])
    imout=cv2.resize(imout,SZ,interpolation=cv2.INTER_AREA)
    
    cv2.imwrite(fileout,imout)
    
        
def time_indicator(filename='time.jpg',fr=15,dt=0.005,nt0=1e5,nt=300,step='log'):
    """
    Generator that creates a jpeg image indicating how much time is elapsed in
    a trajectory for every 1 s real time. Provide a filename for the jpeg (will
    be overwritten at eacy step), the frame rate (fr), the time step (dt), the
    number of time steps in the input (nt0) and the number in the output (nt), 
    and the step mode ('log' or 'linear'). 
    """
#
#    if step=='log':
#        t=(np.logspace(0,np.log10(nt0),nt,endpoint=True)-1)*dt
#    else:
#        t=(np.linspace(0,nt0,nt,endpoint=False))*dt
    
    t=time_axis(fr=fr,nt0=nt0,nt=nt,step='log',mode='avg')
    
    fig=plt.figure(figsize=[4,1]) 
    plt.close(fig)
    ax=fig.add_subplot(111)
    for sp in ax.spines.values():sp.set_color('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    
    fs=r'1 s : {0:d} {1}s'
    
    for k in range(len(t)):
#        i1=np.max([0,k-int(fr/2)])
#        i2=np.min([k+int(fr/2),len(t)-1])
#        Dt=fr*(t[i2]-t[i1])/(i2-i1)
        Dt=t[k]
          
        if np.log10(Dt)<-3:
            Dt*=1e6
            abbr='f'
        elif np.log10(Dt)<0:
            Dt*=1e3
            abbr='p'
        elif np.log10(Dt)<3:
            abbr='n'
        elif np.log10(Dt)<6:
            Dt*=1e-3
            abbr='$\mu$'
        else:
            Dt*=1e-6
            abbr='m'
        Dt=np.round(Dt).astype(int)
        if k==0:
            hdl=ax.text(-.1,.2,fs.format(Dt,abbr),FontSize=35)
        else:
            hdl.set_text(fs.format(Dt,abbr))
            
        fig.savefig(filename)
            
        yield

def text_image(text,filename='text.jpg',figsize=[4,1],FontSize=25):
    """
    Save an image with text.
    
    text_image(text,filename='text.jpg',figsize=[4,1],FontSize=25)
    """
    fig=plt.figure(figsize=figsize) 
    plt.close(fig)
    ax=fig.add_subplot(111)
    for sp in ax.spines.values():sp.set_color('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    
    ax.text(-.1,.2,text,FontSize=FontSize)
    
    fig.savefig(filename)
    
def Ct_plot_updater(ct,filename='ctplot.jpg',fr=15,dt=0.005,nt0=1e5,nt=300,\
                    step='log',titles=None,legends=None,figsize=[5,4],RI=True,\
                    tmode='avg'):
    """
    Plots a correlation function or sets of correlation functions. If ct is 2D,
    then one plot is created, but if it is 3D, then the outer dimension is plotted
    in separate plots. The plots are only shown out to the current time point. 
    One may provide a title and legend (if 2D), or a list of titles and legends
    (if 3D, may use None to omit some legends). Results are not shown: they are
    only stored in the file given by filename
    
    Ct_plot_updater(filename='ctplot.jpg',ct,dt=0.005,nt0=1e5,nt=300,step='log',titles=None,legends=None,figsize=[5,4])
    
    Set RI to True to plot the real part of the second element, the real/imag
    parts of the third element, and the real/imag parts of the fourth element
    (0th and 1st element assumed to be conjugates/negative conjugates of these)
    Add a 6th element to the correlation function: we assume this is the total
    correlation function for a motion and plot in black.
    """

    
    if tmode[:1].lower()=='t' or tmode[:1]=='i':
        t=time_axis(fr=fr,dt=dt,nt0=nt0,nt=nt,step=step,mode='index')
    elif len(tmode)>=3 and tmode[:3].lower()=='avg':
        t=time_axis(fr=fr,dt=dt,nt0=nt0,nt=nt,step=step,mode='avg_index')
    else:
        print('tmode not recognized: set to "avg" or "t"')
        return
    ct=np.array(ct)
    
    fig=plt.figure(figsize=figsize)
    plt.close(fig)
    
    
    "Make sure 3D (middle dimension can have variable size in this setup)"
    if not(hasattr(ct[0],'__len__') and hasattr(ct[0][0],'__len__')): 
        ct=np.moveaxis(np.atleast_3d(ct),-1,0)
        
    "Make sure titles is a list"
    if titles is not None and isinstance(titles,str):
        titles=[titles]
        
    "Make sure legends is a list (of lists)"
    if legends is not None and isinstance(legends[0],str):
        legends=[legends]
    
    npl=len(ct)        
    
    ax=[fig.add_subplot(npl,1,k) for k in range(1,npl+1)]
    
    
    hdl=[]
    
    for k in range(len(t)):
        if k==0:
            for m,a in enumerate(ax):
                h=list()
                if step=='log':
                    if RI:
                        for q in range(2,5):h.append(a.semilogx(np.arange(1,t[k]+1)*dt,ct[m][q,1:t[k]+1].T.real,marker='o',markersize=1)[0])
                        for q in range(3,5):h.append(a.semilogx(np.arange(1,t[k]+1)*dt,ct[m][q,1:t[k]+1].T.imag,marker='o',markersize=1)[0])
                        if len(ct[m])==6:
                            h.append(a.semilogx(np.arange(1,t[k]+1)*dt,ct[m][-1,1:t[k]+1].T.imag,\
                                                marker='o',markersize=1,markerfacecolor='black',color='black')[0])
                    else:
                        clr=[[0.5,.5,.5],'black','#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
                        ls=['--','-',':',':',':',':',':']
                        for n,ct0 in enumerate(ct[m]):
                            
                            h.append(a.semilogx(np.arange(1,t[k]+1)*dt,ct0[1:t[k]+1],markersize=1,color=clr[n],linestyle=ls[n])[0])
#                        h=a.semilogx(np.arange(1,t[k]+1)*dt,ct[m][:,1:t[k]+1].T,marker='o',markersize=1)
                    
                    for q,h0 in enumerate(h):      
                        if RI:
                            if q==5:
                                a.semilogx(dt*np.min(t[t>0])/5,ct[m][-1,0].T.real,Marker='o',\
                                           markerfacecolor=h0.get_color(),markeredgewidth=0)
                            elif q<3:
                                a.semilogx(dt*np.min(t[t>0])/5,ct[m][q+2,0].T.real,Marker='o',\
                                           markerfacecolor=h0.get_color(),markeredgewidth=0)
                            else:
                                a.semilogx(dt*np.min(t[t>0])/5,ct[m][q,0].T.imag,Marker='o',\
                                           markerfacecolor=h0.get_color(),markeredgewidth=0)
                            
                        else:
                            a.semilogx(dt/5,ct[m][q,0].T,Marker='o',markerfacecolor=h0.get_color(),markeredgewidth=0)
                    a.set_xlim(dt/5,t[-1]*dt)
                else:
                    if RI:
                        for q in range(2,5):h.append(a.plot(0,ct[m][q,0].T.real)[0])
                        for q in range(3,5):h.append(a.plot(0,ct[m][q,0].T.imag)[0])
                        if len(ct[m])==6:
                            h.append(a.plot(0,ct[m][-1,0].T.real)[0])
                    else:
                        h=a.plot(0,ct[m][:,0].T)
                        
                    for q,h0 in enumerate(h):      
                        if RI:
                            if q==5:
                                a.plot(0,ct[m][-1,0].T.real,Marker='o',markerfacecolor=h0.get_color(),markeredgewidth=0)
                            elif q<3:
                                a.plot(0,ct[m][q+2,0].T.real,Marker='o',markerfacecolor=h0.get_color(),markeredgewidth=0)
                            else:
                                a.plot(0,ct[m][q,0].T.imag,Marker='o',markerfacecolor=h0.get_color(),markeredgewidth=0)
                        else:
                            a.plot(0,ct[m][q,0].T,Marker='o',markerfacecolor=h0.get_color(),markeredgewidth=0)
                    a.set_xlim(0,t[-1]*dt)
                if RI:     
                    a.set_ylim(np.min([-.1,1.2*ct[m][2:].real.min(),1.2*ct[m][2:].imag.min()]),1.1)
                else:
                    a.set_ylim(np.min([-.1,ct[m].real.min()]),1.1)
                hdl.append(h)
                a.set_ylabel('C(t)')
                if m==len(ax)-1:
                    a.set_xlabel(r'$t$ / ns')
                else:
                    a.set_xticklabels([])
                if legends is not None and legends[m] is not None:
                    a.legend(legends[m],loc='upper right',fontsize=8,fancybox=True,framealpha=.5)
                if titles is not None:
                    a.set_title(titles[m])
            fig.tight_layout()
            fig.savefig(filename)
            yield
        else:
            for m,(a,h) in enumerate(zip(ax,hdl)):
                for q,h0 in enumerate(h):
                    if step=='log':
                        h0.set_xdata(np.arange(1,t[k]+1)*dt)
                        if RI:
                            if q==5:
                                h0.set_ydata(ct[m][-1,1:t[k]+1].real)
                            elif q<3:
                                h0.set_ydata(ct[m][q+2,1:t[k]+1].real)
                            else:
                                h0.set_ydata(ct[m][q,1:t[k]+1].imag)
                        else:
                            h0.set_ydata(ct[m][q,1:t[k]+1])
                    else:
                        h0.set_xdata(np.arange(0,t[k])*dt)
                        if RI:
                            if q==5:
                                h0.set_ydata(ct[m][-1,0:t[k]].real)
                            elif q<3:
                                h0.set_ydata(ct[m][q+2,0:t[k]].real)
                            else:
                                h0.set_ydata(ct[m][q,0:t[k]].imag)
                        else:
                            h0.set_ydata(ct[m][q,0:t[k]])
                fig.savefig(filename)
            yield
                    
                
def rotate_sel(v0,v,pos0,pivot):
    """Calculates the positions of a set of atoms that are rotated due to a frames
    current position. Provide the selections, pivot points, and values of vector
    functions.
    
        v0      :   Values of vZ and vXZ  at time 0 (or whichever time point is
                    being used to construct the frame motion)
        v       :   Current set of vZ and vXZ vectors. (v0 and v are both 2-
                    element tuples which contain vectors with shape 3xN)
        pos0     :  List of positions for each frame. (N-element list with 3xM
                    positions, where M is the number of atoms)
        pivot   :   List of pivot points for each frame. (N-element list with 
                    3-element position)
    
    pos = rotate_sel(v0,v,pos0,pivot)
    """
    
    sc=vft.getFrame(*vft.applyFrame(*v,nuZ_F=v0[0],nuXZ_F=v0[1]))    
    pos=[vft.R(p-pv,*sc0)+pv for (p,pv,sc0) in zip(pos0,pivot,sc)]
    
    return pos
    
def frame_gen(pdb_template,mol,sel0,sel,pivot,order=None,fn=0,nt0=1e5,nt=1e3,step='log',Dxyz=[0,0,0]):
    """
    Writes pdbs where the motion in those pdbs is the result of motion of a specific
    frame. One must provide a file template, a trajectory object, a selection to
    include in the pdbs, a vector function defining the frame being moved (which
    returns N frames), and optionally a vector funtion defining an out frame (
    a frame index may be provided that maps vF, the outer function, to vf, the
    vector function responsible for motion, or a list of two elements that maps
    both functions to the bond), sa list of selections (N selections), and a 
    list of N 1-atom selections that serve as pivot points for each of the frames.
    
    We also require defintion of the time axis, that is, nt0, nt, and step
    
    Note that the pivot point for a frame may lie within another frame. In this
    case, the latter frame in the list may have its center moved by re-positioning
    of the pivot point. Caution should be taken when this is the case, since it 
    adds an apparent motion to the latter frame. However, this is only translational
    motion, so it is reasonable to plot it this way, since such motion has no
    influence on the reorientational correlation function.
    
    One may re-order the frames, to change which motions move which pivots.
    Specify "order" to achieve this. Order is an integer array (list), which 
    specifies which order to apply the frames.
    
    Note that having the pivot point inside another selection may cause the
    appearance of unwanted bond-breaking. This can be resolved by performing:
        sel0.universe.add_TopologyAttr('bonds',[(i1,i1)])
    where i1 and i2 are the indices of the two atoms which are to remain bonded.
    
    Finally note, if the pivot of one frame lies within another frame, then one
    should not include the pivot in the selection. Otherwise we translate the
    pivot incorrectly
    """
    
    
    index=time_axis(nt0=nt0,nt=nt,step=step,mode='index')
    
    "List of indices to find each selection in sel within sel0"
    si=[]
    for s in sel:
        if s is None:
            si.append(None)
        else:
            si.append(np.array([np.argwhere(s0.index==sel0.indices).squeeze() for s0 in s],dtype=int))
    "List of indices to find the pivot points within sel0"
    pi=[None if p0 is None else \
                 np.argwhere(p0.index==sel0.indices).squeeze().astype(int) for p0 in pivot]
    
    
    "Get the frame vectors"    
    v=ef.mol2vec(mol,index=index)
    vZ,vXZ,nuZ,nuXZ,fi=ef.apply_fr_index(v)
    
    if fn==0:
        nuZ_F,nuXZ_F=nuZ[0],nuXZ[0] 
#        fi=np.arange(vZ.shape[1])
    elif fn==len(nuZ):
        vZ,vXZ=nuZ[-1],nuXZ[-1]
        nuZ_F,nuXZ_F=None,None
#        fi=np.unique(v['frame_index'][-1],return_index=True)[1]
#        fi=fi[np.logical_not(np.isnan(v['frame_index'][-1][fi]))]
#        fi=np.unique(fi[-1],return_index=True)[1]
#        fi=fi[fi>=0]
    else:
        vZ,vXZ=nuZ[fn-1],nuXZ[fn-1]
        nuZ_F,nuXZ_F=nuZ[fn],nuXZ[fn]
#        fi=np.unique(fi[fn-1],return_index=True)[1]
#        fi=fi[fi>=0]
#        fi=np.unique(v['frame_index'][fn-1],return_index=True)[1]
#        fi=fi[np.logical_not(np.isnan(v['frame_index'][fn-1][fi]))]
    fi1=np.unique(fi[fn],return_index=True)[1]
    fi=fi1[fi[fn][fi1]>=0]

            
    vZ=vft.norm(vZ[:,fi])
    if vXZ is not None:vXZ=vXZ[:,fi]
    if nuZ_F is not None:nuZ_F=vft.norm(nuZ_F[:,fi])
    if nuXZ_F is not None:nuXZ_F=nuXZ_F[:,fi]
    
    vZfF,vXZfF=vft.applyFrame(vZ,vXZ,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F)

    
    """
    here, we have the initial orientation of the outer frame. Starting positions
    of the bonds should be calculated by removing this rotation, and then
    adding it back in *after* the inner frame's rotation has been performed.
    
    Probably, the inner frame's rotation should also be removed to get the 
    initial positions, since it will be added back at the first time step.]
    
    Application of sc0 below doesn't make all that much sense to me here. 
    Probably this also needs to be removed
    """
    if nuZ_F is not None:
        scF=np.array(vft.getFrame(nuZ_F[:,:,0],nuXZ_F[:,:,0]))
    else:
        scF=np.zeros([6,vZ.shape[1]])
        scF[[0,2,4]]=1
    scF=scF.T
#    vZfF=vft.R(vZfF.swapaxes(1,2),*sc0).swapaxes(1,2)
#    vXZfF=vft.R(vXZfF.swapaxes(1,2),*sc0).swapaxes(1,2)
    
    sc=np.array(vft.getFrame(vZfF,vXZfF)).T

    

    "Reorder the frames (in case pivot points overlap into other frames)"    
    order=np.arange(len(si),dtype=int) if order is None else np.array(order,dtype=int)
    si=[si[o] for o in order]
    pi=[pi[o] for o in order]
    
    
    "Calculate initial positions (go into frame of vf at initial time)"
    mol.mda_object.trajectory[0]
    
    pos0=sel0.positions.copy()
    #    pos0=pos0-pos0.mean(0)+np.array(Dxyz)  #Center the molecule
    pos0=pos0-sel0.positions.mean(0) #Center the molecule
    
    pos00=pos0.copy()
    for si0,pi0,sc0,sc0F in zip(si,pi,sc[0][order],scF[order]):
        if pi0 is not None:
            pos0[si0]+=(pos0[pi0]-pos00[pi0])   #If the pivot point moved, then shift all atoms in the group
            pos0[si0]=vft.R(vft.R((pos0[si0]-pos0[pi0]).T,*vft.pass2act(*sc0F)),*vft.pass2act(*sc0)).T+pos0[pi0]


    

#    sel0.guess_bonds()
    "Sweep over and write out all pdbs"
    for k,sc1 in enumerate(sc):
        pos=pos0.copy()
        for si0,pi0,sc0,sc0F in zip(si,pi,sc1[order],scF[order]):
            if pi0 is not None:
                pos[si0]+=(pos[pi0]-pos0[pi0]) #If the pivot point moved, then shift all atoms in the group
                pos[si0]=vft.R(vft.R((pos[si0]-pos[pi0]).T,*sc0),*sc0F).T+pos[pi0]
        sel0.positions=pos.copy()
        sel0.write(pdb_template.format(k),bonds='all')
#        sel0.write(file_template.format(k))
        
        
        

def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]  

        
        


    


        
        
        
        
        
        
        
        
        
        
    



        








    