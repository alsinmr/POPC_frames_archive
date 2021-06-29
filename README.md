# POPC_frames_archive
This archive provides the various python scripts required to perform the analyses 
shown in the paper "Characterizing the dynamic landscape of a bio-membrane with 
experiment and simulation", 

by A.A. Smith, A. Vogel, O. Enberg, P. Hildebrand, D. Huster

albert.smith-penzel@medizin.uni-leipzig.de

The main script is POPC_full_ana.py, which will execute in sequence the analysis
of NMR experimental data, followed by analysis of MD simulated data for
comparison, including plotting (exp_v_sim.py).

Several external python modules are required (in addition to the standard libraries):

numpy v. 1.17.2,
scipy v. 1.3.0,
pandas v. 0.25.1,
MDAnalysis v. 0.19.2,
matplotlib v. 3.0.3,
cv2 v. 4.1.0 (for Landscape plotting)

Visualization makes use of ChimeraX (v. 1.0), which must be installed on the user's
system (https://www.rbvi.ucsf.edu/chimerax/). The path to the executable must
be provided, by running pyDIFRATE.chimeraX.set_chimera_path(path) (only needs to be run
once).

All files are copyrighted under the GNU General Public License. A copy of the license has been provided in the file LICENSE

Copyright 2021 Albert Smith-Penzel