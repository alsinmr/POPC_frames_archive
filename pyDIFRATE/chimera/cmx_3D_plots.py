#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:58:47 2021

@author: albertsmith
"""

import os
import numpy as np
import cv2
from shutil import copyfile
#import chimera.chimeraX_funs as cmx
from pyDIFRATE.chimera.chimeraX_funs import py_line,chimera_path,get_path,py_print_npa,WrCC


def copy_imports(f):
    """
    Copies all imports required for this library to be used in ChimeraX 
    (may result in some unnecessary imports for the individual functions)
    """
    copy_fun(f,'imports')

def copy_fun(f=None,name=None):
    """
    Some of the functions in this file will be copied into chimera scripts. We
    will tag these in the comments with a starting and ending code as follows. 
    This format must be matched exactly (also- no indent or trailing spaces!)
    
    #%%BEGIN CMX FUNCTION name
    #everything in between gets copied
    #%%END CMX FUNCTION name
    
    name gets replaced by the function name. The function will be copied as it
    appears in this file. Must also pass a file handle to write into the chimera
    script.
    
    Note that we do not check if the in between parts are really a function,
    so you can in principle copy groups of functions in a single call.
    
    copy_fun(f,name)
    
    Calling copy_fun without name will return a list of all functions available
    for copy.
    """
    
    if name is None:
        funs=list()
        with open(get_path('cmx_3D_plots.py'),'r') as f1:
            for line in f1:
                if len(line)>=22 and line[:22]=='#%%BEGIN CMX FUNCTION ':
                    funs.append(line[22:-1])
        return funs
                
    else:
        start_code="#%%BEGIN CMX FUNCTION "+name
        end_code="#%%END CMX FUNCTION "+name
        
        with open(get_path('cmx_3D_plots.py'),'r') as f1:
            start_copy=False
            end_copy=False
            for line in f1:
                if start_copy:
                    if line[:-1]==end_code:
                        end_copy=True
                        break
                    f.write(line)

                else:
                    if line[:-1]==start_code:start_copy=True
            assert start_copy,"Could not find the start_code in file: {0}".format(start_code)
            assert end_copy,"Could not find the end_code in file: {0}".format(end_code)
        f.write('\n')

def make_surf(x,y,z,colors=None,triangles=None,chimera_cmds=None,fileout=None,save_opts=None,scene=None):
    """
    Plots a surface in chimeraX from x,y, and z. May also include colors
    """
    rand_index=np.random.randint(1e6)   #We'll tag a random number onto the filename
                                        #This lets us run multiple instances without interference
    full_path=get_path('chimera_script{0:06d}.py'.format(rand_index))     #Location to write out chimera script
    
    with open(full_path,'w') as f:
        copy_fun(f,'imports')
        copy_fun(f,'surf3D')
        
        py_line(f,'try:')
        py_print_npa(f,'x',x,nt=1)
        py_print_npa(f,'y',y,nt=1)
        py_print_npa(f,'z',z,nt=1)
        if triangles is None:
            py_line(f,'triangles=None',1)
        else:
            py_print_npa(f,'triangles',triangles,format_str='d',dtype='int32',nt=1)
        if colors is None:
            py_line(f,'colors=None',nt=1)
        else:
            if colors.ndim>2:
                for m,c0 in enumerate(colors):
                    py_print_npa(f,'c0{0}'.format(m),c0,format_str='d',dtype='uint8',nt=1)
#                py_line(f,'colors=[locals()["c0{0}".format(k)] for k in range({1})]'.format('{0}',m+1),1)
                py_line(f,'colors=list()',1)
                py_line(f,'keys=["c0{0}".format(m) for m in range({1})]'.format("{0}",m+1),1)
                py_line(f,'for key in keys:',1)
                py_line(f,'colors.append(locals()[key])',2)
            else:
                py_print_npa(f,'colors',colors,nt=1)
        
        if scene:
            WrCC(f,'open '+scene,1)
            
        
        py_line(f,'surf3D(session,x,y,z,colors,triangles)',1)
        
        if chimera_cmds is not None:
            if isinstance(chimera_cmds,str):chimera_cmds=[chimera_cmds]
            for cc in chimera_cmds:
                WrCC(f,cc,1)
                
        if fileout is not None:
            if len(fileout)>=4 and fileout[-4]!='.':fileout=fileout+'.png'
            if save_opts is None:save_opts=''
            WrCC(f,"save " +fileout+' '+save_opts,1)
            
        
        py_line(f,'except:')
        py_line(f,'pass',1)
        py_line(f,'finally:')
        py_line(f,'os.remove("{0}")'.format(full_path),1)
#        py_line(f,'pass',1)
        if fileout is not None: #Exit if a file is saved
            WrCC(f,'exit',1)
#            pass
    copyfile(full_path,full_path[:-9]+'.py')
    
#    if fileout is None:
    os.spawnl(os.P_NOWAIT,chimera_path(),chimera_path(),full_path)
#    else:
#        os.spawnl(os.P_NOWAIT,chimera_path(),chimera_path(),'--nogui '+ full_path)
            
def multi_surf(x,y,z,colors=None,triangles=None,chimera_cmds=None,fileout=None,save_opts=None,scene=None,debug=False):
    """
    Takes lists of x, y, and z coordinates, and optionally also colors. Each
    element of the lists corresponds to one surface to be plotted in chimera.
    """
    rand_index=np.random.randint(1e6)   #We'll tag a random number onto the filename
                                        #This lets us run multiple instances without interference
    full_path=get_path('chimera_script{0:06d}.py'.format(rand_index))     #Location to write out chimera script
    
    if triangles is None:triangles=[None for _ in range(len(x))] #Empty triangles variable
    
    with open(full_path,'w') as f:
        copy_fun(f,'imports')
        copy_fun(f,'surf3D')
        py_line(f,'try:')
        for k,(x0,y0,z0,c0,tri0) in enumerate(zip(x,y,z,colors,triangles)):
            py_print_npa(f,'x{0}'.format(k),x0,nt=1)
            py_print_npa(f,'y{0}'.format(k),y0,nt=1)
            py_print_npa(f,'z{0}'.format(k),z0,nt=1)
            if tri0:
                py_print_npa(f,'tri{0}'.format(k),tri0,format_str='d',dtype='int32',nt=1)
            else:
                py_line(f,'tri{0}=None'.format(k),1)
            if colors is None:
                py_line(f,'colors{0}=None'.format(k),nt=1)
            else:
                if c0.ndim>2:
                    for m,c00 in enumerate(c0):
                        py_print_npa(f,'c0{0}'.format(m),c00,format_str='d',dtype='uint8',nt=1)
                        
#                    py_line(f,'colors{0}=np.array([c0{1}.format(k) for k in range({2})])'.format(k,"{0}",m+1),1)
                    py_line(f,'colors{0}=list()'.format(k),1)
                    py_line(f,'keys=["c0{0}".format(m) for m in range({1})]'.format("{0}",m+1),1)
                    py_line(f,'for key in keys:',1)
                    py_line(f,'colors{0}.append(locals()[key])'.format(k),2)
                else:
                    py_print_npa(f,'colors{0}'.format(k),c0,nt=1)
        if scene:
            WrCC(f,'open '+scene,1)
            
        for k in range(len(x)):
            py_line(f,'surf3D(session,x{0},y{0},z{0},colors{0},tri{0})'.format(k),1)
            
        if chimera_cmds is not None:
            if isinstance(chimera_cmds,str):chimera_cmds=[chimera_cmds]
            for cc in chimera_cmds:
                WrCC(f,cc,1)
                
        if fileout is not None:
            if len(fileout)>=4 and fileout[-4]!='.':fileout=fileout+'.png'
            if save_opts is None:save_opts=''
            WrCC(f,"save " +fileout+' '+save_opts,1)
        py_line(f,'close=True',1)
        py_line(f,'except:')
        py_line(f,'close=False',1)
        py_line(f,'print("ChimeraX script failed")',1)
        WrCC(f,'ui tool show Log',1)
        py_line(f,'finally:')
        py_line(f,'os.remove("{0}")'.format(full_path),1)
#        py_line(f,'pass',1)
        if fileout is not None: #Exit if a file is saved
            py_line(f,'if close:',1)
            WrCC(f,'exit',2)
#            pass
    copyfile(full_path,full_path[:-9]+'.py')
    
    if debug:
        os.system(chimera_path()+' --debug '+full_path)
    else:
        os.spawnl(os.P_NOWAIT,chimera_path(),chimera_path(),full_path)

def image2surf(filename,x0,y0,xLen,yLen=None):
    im=cv2.imread(filename)     
        
"""
The imports should be kept in quotes as such, so that these imports are not
imported into the library here (some imports are only available within 
ChimeraX, and would create errors, besides slowing down the file)
#%%BEGIN CMX FUNCTION imports
import os
import numpy as np
from matplotlib.tri import Triangulation
from chimerax.core.models import Surface
from chimerax.surface import calculate_vertex_normals,combine_geometry_vntc
from chimerax.core.commands import run as rc
#%%END CMX FUNCTION imports
"""

#%%BEGIN CMX FUNCTION all_funs
#%%BEGIN CMX FUNCTION surf3D
def surf3D(session,x,y,z,colors=None,triangles=None):
    """
    Creates a 3D surface plot in chimera. Input are the x,y, and z coordinates
    (x and y can be the same size as z, or may be x and y axes for the z plot, 
    in which case x.size==z.shape[0], y.size==z.shape[1].)
    
    Colors may also be specified for each vertex (should then have 3xN or 4xN
    points, where N=z.size), or a single color may be specified for the whole 
    plot. 4xN allows specification of the opacity. Us 0-255 RGB specification.
    """
    
    if colors is None:
        colors=np.array([[210,180,140,255]]).T.repeat(z.size,axis=1)
    else:
        colors=np.array(colors)
        if colors.size==3:colors=np.concatenate((colors,[255]))
        if colors.size==4:colors=np.atleast_2d(colors).T.repeat(z.size,axis=1)
        if colors.ndim>2:colors=np.reshape(colors,[colors.shape[0],colors.shape[1]*colors.shape[2]])
    
    if not(x.size==z.size and y.size==z.size):
        x,y=[q.reshape(x.size*y.size) for q in np.meshgrid(x,y)]
    if triangles is None:
        triangles=Triangulation(x,y).triangles
    z=z.reshape(z.size)
    
    xyz=np.ascontiguousarray(np.array([x,y,z]).T,np.float32)       #ascontiguousarray forces a transpose in memory- not just editing the stride
    colors = np.array(colors, np.uint8).T
    tri = np.array(triangles, np.int32)
    
    norm_vecs=calculate_vertex_normals(xyz,tri)
    
    s=Surface('surface',session)
    s.set_geometry(xyz,norm_vecs,tri)
    s.vertex_colors=colors
    session.models.add([s])

#%%END CMX FUNCTION surf3D
    
#%%BEGIN CMX FUNCTION sphere_triangles
def sphere_triangles(theta_steps=100,phi_steps=50):
    """
    Creates arrays of theta and phi angles for plotting spherical tensors in ChimeraX.
    Also returns the corresponding triangles for creating the surfaces
    """
    
    theta=np.linspace(0,2*np.pi,theta_steps,endpoint=False).repeat(phi_steps)
    phi=np.repeat([np.linspace(0,np.pi,phi_steps,endpoint=True)],theta_steps,axis=0).reshape(theta_steps*phi_steps)
    
    triangles = []
    for t in range(theta_steps):
        for p in range(phi_steps-1):
            i = t*phi_steps + p
            t1 = (t+1)%theta_steps
            i1 = t1*phi_steps + p
            triangles.append((i,i+1,i1+1))
            triangles.append((i,i1+1,i1))
    
    return theta,phi,triangles
#%%END CMX FUNCTION sphere_triangles

#%% END CMX FUNCTION all_funs

#%% Delete functions intended only for chimeraX
"""It doesn't make sense to have these availabe outside of chimeraX, since usually
they won't run"""
del(surf3D)
   
#%% Other cleanup- things we only need internally 
#del(os)
#del(copyfile)
#del(np)

