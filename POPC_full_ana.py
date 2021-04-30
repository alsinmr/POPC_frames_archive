#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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


import runpy.runpath as run
import DIFRATE as DR


"First, we analysis the experimental data. Results are stored in fit_exper"
run('exp_ana.py')

"""Second, we do a pre-analysis of the simulated data. That is, we fit a large
number of detectors to the MD-derived correlation functions (20). However, these
are not optimized to have ideal shapes (direct output of the SVD). These may
be re-processed later to match the experimental detector sensitivities.
"""
#run('sim_ini_ana.py')
"""This has already been run, and the output provided in data_sim folder. 
Please request MD data directly if you want to re-run this!
(the file checks for the existence of the output, and will not attempt to 
re-calculate the existing data)
"""

"Next, we compare the experimental and simulated results"
run('exp_v_sim.py')

"We combine experimental and simulated analysis to make 3D plots of detector responses"
#This will only work if you've provided the path to chimeraX, use line below
#DR.chimeraX.set_chimera_path(path)
run('bonbon_plots.py')

