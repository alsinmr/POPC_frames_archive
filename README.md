# POPC_frames_archive
This archive provides the various python scripts required to perform the analyses 
shown in the paper "Characterizing the dynamic landscape of a bio-membrane with 
experiment and simulation", 

by A.A. Smith, A. Vogel, O. Enberg, P. Hildebrand, D. Huster

The main script is POPC_full_ana.py, which will execute in sequence the analysis
of NMR experimental data, followed by analysis of MD simulated data for
comparison, including plotting (exp_v_sim.py).

Several external python modules are required (in addition to the standard libraries)
numpy
scipy
MDAnalysis
matplotlib

Some visualization makes use of ChimeraX, which must be installed on the user's
system (https://www.rbvi.ucsf.edu/chimerax/). The path to the executable must
be provided, by running chimeraX.set_chimera_path(path) (only needs to be run
once).