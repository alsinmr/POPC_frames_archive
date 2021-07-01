#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2021 Albert Smith-Penzel

This file is part of POPC frames archive (PFA).

PFA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PFA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PFA.  If not, see <https://www.gnu.org/licenses/>.


Questions, contact me at:
albert.smith-penzel@medizin.uni-leipzig.de


Created on Thu Apr 29 12:38:43 2021

@author: albertsmith
"""


"""
For our paper, "Characterizing the dynamic landscape of a bio-membrane with 
experiment and simulation", we want to provide a thorough description of how
the various calculations are performed. We do so here by providing the code 
required for creating each figure.

For the various analyses, the initial computation requires having the full MD
trajectory. For sake of saving space on the GitHub archive, we provide the 
required code, but do not include the trajectory itself. The initial step of 
analysis reduces the data size significantly, and so we provide the result of 
the initial analysis (and comment out the lines responsible for performing
the initial analysis). MD trajectory is available upon request
"""


from runpy import run_path as run


"First, we analysis the experimental data. Results are stored in fit_exper"
_=run('exp_ana.py')  

"""Second, we do a pre-analysis of the simulated data. That is, we fit a large
number of detectors to the MD-derived correlation functions (20). However, these
are not optimized to have ideal shapes (direct output of the SVD). These may
be re-processed later to match the experimental detector sensitivities.
"""
#_=run('sim_ini_ana.py')

"""This has already been run, and the output provided in data_sim folder. 
Please request MD data directly if you want to re-run this!
(the file checks for the existence of the output, and will not attempt to 
re-calculate the existing data)
"""

"Next, we compare the experimental and simulated results"
_=run('exp_v_sim.py')

"We combine experimental and simulated analysis to make 3D plots of detector responses"
#This will only work if you've provided the path to chimeraX, use line below
#DR.chimeraX.set_chimera_path(path)
from bonbon_plots import Bonbon
for k in range(4):Bonbon(k)

"""Next, we do pre-analysis of the frame data 
(Defining of frames, loading frames, processing with un-optimized detectors,
averaging of results)
This has been run, and the averages already stored in folder data_frames
"""
#_=run('frame_ini_ana.py')


"""We process the pre-analysis of frames, and plot the detector responses for
the detectors matched to the experimental detector responses"""
_=run('frame_ana.py')

"""We create the bonbon plots for the full frame analysis. Change the frame number
in the following function to switch the frame being plotted (launches four
instances of chimeraX). As with the previous bonbon plots, you must first setup
chimeraX on your system
"""
from bonbon_frames import Bonbon_fr
Bonbon_fr(3)   #Select which frame to plot with the index here (0-3)
#for k in range(4):Bonbon(k)     #Or plot all frames: will make 16 instances of ChimeraX!!

"""We extract the residual tensors for one copy of POPC. The first step requires
having the full trajectories. Here, we have saved the results, and then plot
the results in the second step
"""
#import residual tensors  #Calculate the residual tensors (only done if you have trajectorys)

from Tensor_plots import plot_tensor
for k in range(4):plot_tensor(k)     #plot all residual tensors (4 instances of chimeraX)


"""We fit parameters to describe the dynamics landscape, first using only MD
data (later, we refine the results with NMR data)
"""
_=run('LS_MD_only.py')

"""Now we refine the fit to better match the experimental results. This is done
by varying the corellation time of internal motion. Where multiple residues
overlap in the spectrum, or where there are two internal motions, we scale all
relevant correlation times by the same factor (we actually operate on the log-
correlation time, so we add the same factor to all log-correlation times)
"""
_=run('LS_NMR_refinement.py')

"""Finally, we plot the resulting dynamic landscape. Note that in-text figures
have had axes added in Adobe Illustrator. Here, the POPC molecule is displayed, 
and the timescale axis runs from 10 fs to 1 microsecond."""

from LS_draw import draw_landscape
"""Plots the landscape. The first argument is the frame index (fr_index), which 
may be set to None to obtain the product of all frames, or from 0-3 to select
a particular frame. The second argument is the mode, set to 'NMR' or 'MD'. 
Setting to 'NMR' will plot the experimentally-refined landscapes, whereas MD will
plot the results based on MD data only.
"""
draw_landscape(None,'NMR')
