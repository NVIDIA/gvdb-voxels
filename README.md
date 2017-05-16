
NVIDIA(R) GVDB VOXELS
Release 1.0

OVERVIEW
============
NVIDIA GVDB is a new library and SDK for compute, simulation and rendering of 
sparse volumetric data. Details on the GVDB technology can be 
found at: 
   http://developer.nvidia.com/gvdb

RELEASE NOTES
=============
5/1/2017, GVDB Voxels, Release 1.0 
Created: Rama Hoetzlein, 2017
- First public release
- Open source (BSD 3-clause)
- Samples: g3DPrint, gFluidSurface, gInteractiveGL, gInteractiveOptix,
   gJetsonTX, gRenderKernel, gRenderToFile, gSprayDeposit
- Builds on Windows and Linux
- Runs on Quadro, GeForce, JetsonTX1/2, and Tegra/GRID


REQUIREMENTS
============
  NVIDIA Kepler generation or later GPU
  Windows 7, 8, 10 64-bit
  Microsoft Visual Studio 2010 or higher
  CUDA Toolkit 6.5 or higher (Toolkit 8.0 recommended)
  CMake-GUI 2.7 or later
  OptiX 3.9.0 or later (InteractivOptix sample only, download from NVIDIA)

GVDB is released as a library with samples. 
The library and each sample is built separately, using cmake.

WHAT'S IN THE PACKAGE?
======================
	
   - GVDB VOXELS Library
   - Code Samples
	See the included GVDB_Samples_Description.pdf for detailed sample descriptions.
	- gRenderToFile     - Renders a sparse volume to a file using GVDB
	- gRenderToKernel   - Renders a sparse volume using a custom user-kernel
	- gInteractiveGL    - Interactive rendering of a volume using GVDB and OpenGL
	- gInteractiveOptiX - Interactive rendering of a volume and a polygonal model, with poly-to-poly and poly-to-voxel interactions.
	- g3DPrint          - Demonstrates generating cross section slices for 3D printing from a polygonal model
	- gSprayDeposit     - Demostrates simulated spray deposition onto a 3D part
	- gFluidSim         - Demostrates a dynamic simulation with surface rendering by GVDB
        - gJetsonTX         - Simple 3D Printing Driver for the JetsonTX1/2 with volume slicing on Tegra chip
   - GVDB VBX File Specfication
   - GVDB Sample Descriptions
   - GVDB Programming Guide

SAMPLES USAGE
=============
All interactive samples use the following user input interface
   Camera rotation -> move mouse
   Change orientation -> left mouse click
   Zoom -> right mouse click
   Panning -> hold middle button 
A few samples have on-screen GUIs with features that can be toggled by clicking on them.


WINDOWS - QUICK INSTALLATION
============================

Instructions:

1. Unpackage GVDB and samples
    a. Unzip the GVDB SDK package to \gvdb\source

2. Install dependencies
    a. Install cmake-gui 2.7 or later
    b. Install CUDA Toolkit 8.0
    c. Install OptiX 3.9.0 or later (for gInteractiveOptix sample)

3. Build gvdb_library
    a. Run cmake-gui.
        Where is source code: /gvdb/source/gvdb_library
        Where to build bins:  /gvdb/build/gvdb_library
    b. Click Configure to prepare gvdb_library
    c. Click Generate
    d. Open /gvdb/build/gvdb_library/gvdb_library.sln in VS2010/2013
    e. Build the solution in Debug or Release mode.
       * For whichever mode, you must build later samples with same build type.
    f. The gvdb_library must be built prior to running cmake for any sample.

4. Build any sample, e.g. g3DPrint
    a. Run cmake-gui.
        Where is source code: /gvdb/source/g3DPrint
        Where to build bins:  /gvdb/build/g3DPrint
    b. Click Configure to prepare g3DPrint
    c. You should see that cmake locates the GVDB Library paths automatically
       * Specify any paths that cmake indicated are needed       
    d. Click Generate
    e. Open /gvdb/build/g3DPrint/g3DPrint.sln in VS2010/2013
    f. Build the solution

5. Run the sample!
    a. Select g3DPrint as the start up project.
    b. Click run/debug        

LINUX - QUICK INSTALLATION
==========================

Instructions: 

1. Pre-requisites
  1. Install CMake
      sudo apt-get install cmake-qt-gui
  2. Install the CUDA Toolkit 7.5 or later
      sudo ./cuda_7.5.18_linux.run
      Must be done first, before you install NVIDIA drivers
  3. Install the NVIDIA R367 drivers or later
      These can be downloaded from the NVIDIA website
  4. Remove the symoblic libGL, which may incorrectly point to the libGL mesa driver.
      sudo rm -rf /usr/lib/x86_64-linux-gnu/libGL.so
  5. Link the libGL to the NVIDIA driver
      sudo ln -s /usr/lib/nvidia-367/libGL.so /usr/lib/x86_64-linux-gnu/libGL.so
  6. Install additional developer dependencies.
      sudo apt-get install libxinerama-dev
      sudo apt-get install libxrandr-dev
      sudo apt-get install libxcursor-dev
      sudo apt-get install libxi-dev
      sudo apt-get install libx11-dev

2. Install OptiX [optional, for gInteractiveOptiX sample only]
   - OptiX is distributed as a .sh file, which extracts to the current dir.
   - Create a directory for optix in /usr/lib and move the package there before extracting.
   - $ sudo mkdir /usr/lib/optix
   - $ sudo mv NVIDIA-OptiX-SDK-4.0.1-linux64.sh /usr/lib/optix
   - $ cd /usr/lib/optix
   - $ sudo ./NVIDIA-OptiX-SDK-4.0.1.-linux64.sh

3. Set LD_LIBRARY_PATH in bashrc
     a. Open .bashrc. For example: $ emacs ~/.bashrc
     b. Add the following at the end:
           export LD_LIBRARY_PATH=/usr/local/gvdb/lib:/usr/lib/optix/lib64
           * The first path should be the location of libgvdb.so (once installed)
           * The second path should be the location of optix.so
     c. Source the bash (re-run it)
          $ source ~/.bashrc

4. Build the GVDB Library
     a. Unpackage the source tar.gz file
     b. mkdir ~/codes/build/gvdb_library   # make a folder for the build
     c. cmake-gui                          # run cmake-gui with the following settings:
         i.  source: ~/codes/source/gvdb_library
         ii. build:  ~/codes/build/gvdb_library
         iii. Click Configure, and the nGenerate
         iv. cd ~/codes/build/gvdb_library
         v.  sudo make  
         vi. sudo make install             # default install is to /usr/local/gvdb

5. Build a specific Sample
     a. Unpackage the source tar.gz file
     b. mkdir ~/codes/build/g3DPrint       # make a folder for the build
     c. cmake-gui                          # run cmake-gui with the following settings:
         i.  source: ~/codes/source/g3DPrint
         ii. build:  ~/codes/build/g3DPrint
         iii. Click Configure, and the nGenerate
              * Note: If GVDB is not found, set the GVDB_ROOT_DIR to /usr/local/gvdb
                or your preferred gvdb install location from step 4. 
         iv. cd ~/codes/build/g3DPrint
         v.  make
         vi. make install                  # remember to do 'make install' to get all files
     d. Run the sample!
          ./g3DPrint


License 
==========================
BSD 3-clause. Please refer to License.txt


2017 (c) NVIDIA
