# POPC_frames_archive
This archive provides the various python scripts required to perform the analyses 
shown in the paper "Characterizing the dynamic landscape of a bio-membrane with 
experiment and simulation", 

by A.A. Smith, A. Vogel, O. Enberg, P. Hildebrand, D. Huster

albert.smith-penzel@medizin.uni-leipzig.de

The main script is POPC_full_ana.py, which will execute in sequence the analysis
of NMR experimental data, followed by analysis of MD simulated data for
comparison, including plotting (exp_v_sim.py).

Note 1: this script will pause after creation of figures. Close the figures to continue!
Note 2: this script will create 20 figures and 21 instances of ChimeraX. All visual output of this script is stored in the folders additional_files/figs_line**, where the folder name indicates the corresponding line number.
Note 3: On a 2018 Mac mini, POPC_full_ana takes approximately 35 s to run.

There is NO INSTALLATION required for the code. Just place everything in a folder, navigate there, and run. However, python3 and the following modules must be installed from other sources (these are the tested versions, although other versions may work).

Python v. 3.7.3
numpy v. 1.17.2,
scipy v. 1.3.0,
pandas v. 0.25.1,
MDAnalysis v. 0.19.2,
matplotlib v. 3.0.3,
cv2 v. 4.1.0 (only for Landscape plotting)
ChimeraX v. 1.0

We recommend installing Anaconda: https://docs.continuum.io/anaconda/install/
The Anaconda installation includes numpy, scipy, pandas, and matplotlib. 

MDAnalysis is installed by running:
conda config --add channels conda-forge
conda install mdanalysis
(https://www.mdanalysis.org/pages/installation_quick_start/)

cv2 can be installed by running 
conda install -c conda-forge opencv
(https://anaconda.org/conda-forge/opencv)
(The cv2 requirement is only for lines 34 and 172-185 of LS_draw. If you don't want to install cv2, just comment these line. You will then not be able to see the parts of the POPC molecule plotted with the dynamic landscape)


Additionally, for 3D visualization, we use ChimeraX. In order to use the 3D visualization, this must also be installed (https://www.rbvi.ucsf.edu/chimerax/download.html). The path to the executable must then be provided to pyDIFRATE, see POPC_full_ana.py line 71 (DR.chimeraX.set_chimera_path(path), replace path with the full path to the executable)


All files are copyrighted under the GNU General Public License. A copy of the license has been provided in the file LICENSE

Copyright 2021 Albert Smith-Penzel