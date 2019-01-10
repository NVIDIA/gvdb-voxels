
*Updated* 3/14/2018

NVIDIA GVDB Voxels 1.1 
Beta Release Program

OVERVIEW
============
NVIDIA GVDB is a new library and SDK for compute, simulation and rendering of sparse volumetric data. Details on the GVDB technology can be found at: http://developer.nvidia.com/gvdb
This Beta Release program shares GVDB and several samples with invited NVIDIA partners.


RELEASE NOTES
=============

3/14/2018, GVDB 1.1 Beta
- All limitations given in Programming Guide 1.0 removed
- Dynamic topology on GPU
- Multiple contexts and GVDB objects
- Multiple samples with OptiX support
- Improved compute and raytrace performance
- Render from any channel
- Resampling functions
- Grid transforms for rendering

12/6/2016, GVDB 1.0 Beta Release 3
- Improved build system for Windows and Linux
- Rendering using central rayCast function
- Function pointers for render specialization
- Fixes to trilinear surface rendering
- Added support for depth buffer integration (with GL)
- Improvements to color channels

10/25/2016, GVDB 1.0 Beta Release 2
- GVDB Library build on Windows & Linux 
- GVDB Samples build on Windows & Linux
- Linux install instructions (See below)
- Render to multiple render bufs (see g3DPrint)
- g3DPrint sample made interactive
- Dynamic atlas reallocation (see gFluidSim)
- Improved Insert/SplatPoints interface
- gImportVDB not yet updated

9/29/2016, GVDB 1.0 Beta Release 1
- Windows samples
- No linux build


REQUIREMENTS
============

  NVIDIA Kepler generation or later GPU
  Windows 7, 8, 10 64-bit
  Microsoft Visual Studio 2015 (recommended)
  CUDA Toolkit 7.5 or higher
  CMake-GUI 2.7 or later
  OptiX 3.9.0 or later (InteractivOptix sample only, download from NVIDIA)
  OpenVBD 4.0.0 for Windows (ImportVBD code sample only, available online as win_openvdb)

GVDB is released as a library with samples. 
The library and each sample is built separately, using cmake.


WHAT'S IN THE PACKAGE?
======================
	
   - GVDB API Library
   - Code Samples
	See the included GVDB_Samples_Description.pdf for detailed sample descriptions.
	- g3DPrint          - Demonstrates generating cross section slices for 3D printing from a polygonal model
	- gDepthMap         - Show how to merge GVDB volumes and OpenGL polygons using depth buffers
	- gImportVDB        - Loads and renders a sample OpenVDB file into GVDB (requires OpenVDB 4.0.0)
	- gInteractiveGL    - Interactive rendering of a volume using GVDB Voxels
	- gInteractiveOptiX - Interactive rendering of a volume and a polygonal model, with poly-to-poly and poly-to-voxel interactions.
	- gJetsonTX         - Demonstrates running GVDB Voxels on the NVIDIA Jetson TX 
	- gPointCloud       - Demonstrates high quality rendering of point clouds as a volume level-set with OptiX
	- gPointFusion	    - Demonstrates real-time point cloud fusion by simulating LIDAR range data
	- gRenderToKernel   - Renders a sparse volume using a custom user-kernel
	- gRenderToFile     - Renders a sparse volume to a file using GVDB
	- gResample         - Demonstrates loading a dense RAW medical data and conversion to sparse GVDB volume       
	- gSprayDeposit     - Demostrates simulated spray deposition onto a 3D part

   - GVDB VBX File Specfication
   - GVDB Sample Descriptions


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
    b. Install CUDA Toolkit 8.0 or later
    c. Install OptiX 3.9.0 or later (for gInteractiveOptix sample)

3. Build gvdb_library
    a. Unzip the package or clone the git repository
    a. Run cmake-gui.
        Where is source code: /gvdb/source/gvdb_library
        Where to build bins:  /gvdb/build/gvdb_library
    b. Click Configure to prepare gvdb_library. (Select Visual Studio 2015 x64, recommended)
    c. Click Generate
    d. Open /gvdb/build/gvdb_library/gvdb_library.sln in VS2015
    e. Build the solution in Debug or Release mode.
       * For whichever mode, you must build later samples with same build type.
    f. The gvdb_library must be built prior to running cmake for any sample.

4. Build any sample, e.g. g3DPrint
    a. Run cmake-gui.
        Where is source code: /gvdb/source/g3DPrint
        Where to build bins:  /gvdb/build/g3DPrint
    b. Click Configure to prepare g3DPrint. (Select the same compiler as used for gvdb_library)
    c. You should see that cmake locates the GVDB Library paths automatically
       * Specify any paths that cmake indicated are needed       
    d. Click Generate
    e. Open /gvdb/build/g3DPrint/g3DPrint.sln in VS2015
    f. Build the solution

5. Run the sample!
    a. Select g3DPrint as the start up project.
    b. Click run/debug        

LINUX - QUICK INSTALLATION
==========================

Instructions: 


1. Pre-requisites
    a. Install CMake
          sudo apt-get install cmake-qt-gui
    b. Install the CUDA Toolkit 7.5 or later
          sudo ./cuda_7.5.18_linux.run
          * Must be done first, before you install NVIDIA drivers
    c. Install the NVIDIA R367 drivers or later 
          * These can be downloaded from the NVIDIA website
    d. Remove the symoblic libGL, which may incorrectly point to the libGL mesa driver.
          sudo rm -rf /usr/lib/x86_64-linux-gnu/libGL.so
    e. Link the libGL to the NVIDIA driver
          sudo ln -s /usr/lib/nvidia-367/libGL.so /usr/lib/x86_64-linux-gnu/libGL.so
    f. Install additional developer dependencies:
          sudo apt-get install libxinerama-dev
          sudo apt-get install libxrandr-dev
          sudo apt-get install libxcursor-dev
          sudo apt-get install libxi-dev
          sudo apt-get install libx11-dev

2. Install OptiX [optional, for gInteractiveOptiX sample only]
      * OptiX is distributed as a .sh file, which extracts to the current dir.
      * Create a directory for optix in /usr/lib and move the package there before extracting.
      $ sudo mkdir /usr/lib/optix
      $ sudo mv NVIDIA-OptiX-SDK-4.0.1-linux64.sh /usr/lib/optix
      $ cd /usr/lib/optix
      $ sudo ./NVIDIA-OptiX-SDK-4.0.1.-linux64.sh

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
         - source: ~/codes/source/gvdb_library
         - build:  ~/codes/build/gvdb_library
         - Click Configure, and then Generate
     d. Open terminal and goto:
         - >cd ~/codes/build/gvdb_library
         - >sudo make  
         - >sudo make install             # default install is to /usr/local/gvdb

5. Build a specific Sample
     a. Unpackage the source tar.gz file
     b. mkdir ~/codes/build/g3DPrint       # make a folder for the build
     c. cmake-gui                          # run cmake-gui with the following settings:
         - source: ~/codes/source/g3DPrint
         - build:  ~/codes/build/g3DPrint
         - Click Configure, and the nGenerate
              * Note: If GVDB is not found, set the GVDB_ROOT_DIR to /usr/local/gvdb
                or your preferred gvdb install location from step 4. 
     d. Open terminal and goto:
         - >cd ~/codes/build/g3DPrint
         - >make
         - >make install                  # remember to do 'make install' to get all files
     d. Run the sample!
          ./g3DPrint



