#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 13:10:03 2021

@author: albertsmith
"""

import numpy as np
import pyDIFRATE as DR
import os


#%% Analyze experiment results
"""
Analysis of experimental data. Each spin system requires different treatment,
although resulting data may be treated equivalently, with the exception of C=O
data, which has a different set of detector sensitivities. Then, experimental
data is initiall stored by spin system type: relaxation due to CSA (C=O only),
relaxtion due to 1 proton, 2 protons, or 3 protons.
"""
data=list()     #List to store data from each of 4 spin system types
fit0=list()     #List to store the fits from each of 4 spin system types
folder='data_exper'
files=['CSA.txt','1xH.txt','2xH.txt','3xH.txt'] #File locations for each spin system

if os.path.exists('fit_exper'):
    fit=DR.io.load_DIFRATE('fit_exper')
else:
    for f in files:
        """Data is stored an instance of the data class, which contains both the data
        itself, but also information about the experiments used to acquire that data,
        including their sensitivities (data.sens), and instructions on how to fit
        that data using detectors (data.detect)
        """
        data.append(DR.io.load_NMR(os.path.join(folder,f)))  #Load data from text file.
    
        """
        data.detect.r_auto and data.detect.r_target are functions that optimize
        detectors which may be used to fit the given data set. 
        """
        if f=='CSA.txt':
            """This is the carbonyl data. We are missing NOE data for this set, 
            because it requires a bonded H. Therefore, we must optimize detectors 
            for this data set differently than for the rest of the data.
            """
            data[-1].detect.r_auto(4,inclS2=True,NT='MP')
            """We provide the number of detectors (4), plus whether or not we want to
            include DIPSHIFT data (inclS2=True) in our analysis. NT is the
            normalization type, here set to be "MP", or have the maximum (M) of each
            detector sensitivity set to one, and requiring the rho0 detector to be 
            positive (P). Other options are "M", which has the maximum set to one, 
            but rho0 becomes negative so detectors always sum to 1, or "I" which
            makes each detector sensitivity integrate to 1.
            """
        elif f=='1xH.txt':
            """Here, we optimize detectors for all other data. We use 5 detectors
            plus one from the DIPSHIFT analysis, since we have more data than for
            carbonyls.
            """
            data[-1].detect.r_auto(5,inclS2=True,NT='MP')
            target=data[-1].detect.rhoz()[1:]   #Store the detector sensitivities
        else:
            """Detectors for carbons bonded to 2 and 3 H should be identical to those
            optimized for 1H. Therefore, we set these sensitivities to match, 
            using the data.detect.r_target function. Note that r_target includes
            DIPSHIFT data after initial optimization, therefore, we have to exclude
            rho0 from the r_target argument (see line 58 above, where we have removed
            the first detector sensitivity)
            """
            data[-1].detect.r_target(target,inclS2=True,Normalization='MP')
        "Fit the current dtaa set"
        fit0.append(data[-1].fit())
        """This is a little messy below: we have one fewer detector for C=O data. 
        Here, we insert columns filled with zeros so that we can concatenate the
        whole data set into a single object. 
        """
        if f=='CSA.txt':
            flds=['R','R_std','R_u','R_l']
            for fld in flds:
                val0=getattr(fit0[0],fld)
                val=np.concatenate((val0[:,:1],[[0]],val0[:,1:]),axis=1)
                setattr(fit0[0],fld,val)
            flds=['R','R_std','R_u','R_l','Rin','Rin_std']
            fit0[0].S2in=np.array([0])
            fit0[0].S2in_std=np.array([0.01])
                            
    
    
    "Indices to resort the data into an order corresponding to position in the molecule"
    index=[[6],[4,13,14],[1,2,3,5,7,8,9,10,11,12,15,16],[0,17]]
    index=[np.array(i,dtype=int) for i in index]
    
    ind=list()
    for i in index:
        ind.extend(i)
    fit=DR.tools.append_data(fit0,index=ind)    #Tool to support appending of data objects
    "Causes a warning– this can be ignored"
    fit.sens=fit0[1].sens       #Copy the sensitivity for the 1xH
    """Technically, for the carbonyls, we now have the wrong sensitivity, although 
    this is not a problem, since we won't use it anywhere"""
    
    fit.save("fit_exper")
    "We need the following sensitivities later for comparison to simulation"
    DR.io.save_DIFRATE('exper_sens',fit.sens.rhoz())
    DR.io.save_DIFRATE('CSA_exper_sens',fit0[0].sens.rhoz())


