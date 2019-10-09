
NVIDIA(R) GVDB VOXELS
Release 1.1

## OVERVIEW
NVIDIA GVDB is a new library and SDK for compute, simulation and rendering of
sparse volumetric data. Details on the GVDB technology can be
found at:
http://developer.nvidia.com/gvdb

## RELEASE NOTES

3/25/2018, GVDB Voxels 1.1
- Dynamic topology on GPU
- Multiple contexts and GVDB objects
- Multiple samples with OptiX support
- Improved compute and raytrace performance
- Render from any channel
- Resampling functions
- Grid transforms for rendering
- All limitations in Programming Guide 1.0 addressed

9/19/2017, GVDB Voxels, Incremental fix
- Fixed Depth map integrated raycasting
- Fixed Resample function
- New gDepthMap sample, shows how to render OpenGL and GVDB volumes with proper depth integration.
- New gResample sample, shows how to load a dense RAW file into GVDB as sparse data.
- RAW data courtesy of the [Visible Human Project, National Library of Medicine, National Institutes of Health, USA](https://www.nlm.nih.gov/research/visible/getting_data.html)

5/1/2017, GVDB Voxels, Release 1.0
Created: Rama Hoetzlein, 2017
  - First public release
- Open source (BSD 3-clause)
  - Samples: g3DPrint, gFluidSurface, gInteractiveGL, gInteractiveOptix,
  gJetsonTX, gRenderKernel, gRenderToFile, gSprayDeposit
  - Builds on Windows and Linux
  - Runs on Quadro, GeForce, JetsonTX1/2, and Tegra/GRID


## REQUIREMENTS
  NVIDIA Kepler generation or later GPU
  Windows 7, 8, 10 64-bit
  Microsoft Visual Studio 2010 or higher
CUDA Toolkit 6.5 or higher (Toolkit 8.0 recommended)
  CMake-GUI 2.7 or later
OptiX 3.9.0 or later (InteractivOptix sample only, download from NVIDIA)

  GVDB is released as a library with samples.
  The library and each sample is built separately, using cmake.

## WHAT'S IN THE PACKAGE?

  - GVDB VOXELS Library
  - Code Samples<br />
  See the included GVDB_Samples_Description.pdf for detailed sample descriptions.<br />
  * gRenderToFile     - Renders a sparse volume to a file using GVDB
  * gRenderToKernel   - Renders a sparse volume using a custom user-kernel
  * gInteractiveGL    - Interactive rendering of a volume using GVDB and OpenGL
  * gInteractiveOptiX - Interactive rendering of a volume and a polygonal model, with poly-to-poly and poly-to-voxel interactions.
  * g3DPrint          - Demonstrates generating cross section slices for 3D printing from a polygonal model
  * gSprayDeposit     - Demostrates simulated spray deposition onto a 3D part
* gFluidSurface     - Demostrates a dynamic simulation with surface rendering by GVDB (also point clouds from CPU)
  * gJetsonTX         - Simple 3D Printing Driver for the JetsonTX1/2 with volume slicing on Tegra chip
  - GVDB VBX File Specfication
  - GVDB Sample Descriptions
  - GVDB Programming Guide

## SAMPLE USAGE
  All interactive samples use the following user input interface
  Camera rotation -> move mouse
  Change orientation -> left mouse click
  Zoom -> right mouse click
  Panning -> hold middle button
  A few samples have on-screen GUIs with features that can be toggled by clicking on them.

## WINDOWS - QUICK INSTALLATION

### Install dependencies
  1. Install cmake-gui 2.7 or later
  2. Install CUDA Toolkit 8.0
3. Install OptiX 3.9.0 or later (for gInteractiveOptix sample)

### Build GVDB Library
  4. Unzip the package or clone the git repository
  5. Run cmake-gui.
  - Where is source code: /gvdb/source/gvdb_library
  - Where to build bins:  /gvdb/build/gvdb_library
  - Click Configure to prepare gvdb_library
  - Click Generate
  - Open /gvdb/build/gvdb_library/gvdb_library.sln in VS2010/2013
  - Build the solution in Debug or Release mode.
  - For whichever mode, you must build later samples with same build type.
  - The gvdb_library must be built prior to running cmake for any sample.

### Build sample(s)
  6. Run cmake-gui.
  - Where is source code: /gvdb/source/g3DPrint
  - Where to build bins:  /gvdb/build/g3DPrint
  - Click Configure to prepare g3DPrint
  - You should see that cmake locates the GVDB Library paths automatically
  - Specify any paths that cmake indicated are needed
  7. Click Generate
  8. Open /gvdb/build/g3DPrint/g3DPrint.sln in VS2010/2013
  9. Build the solution
  10. Run the sample! Select g3DPrint as the start up project. Click run/debug

## LINUX - QUICK INSTALLATION

### Install Pre-requisites
  1. Install CMake
  - sudo apt-get install cmake-qt-gui
  2. Install the CUDA Toolkit 7.5 or later
  3. Install the NVIDIA R367 drivers or later
  - These can be downloaded from the NVIDIA website
  4. Remove the symbolic libGL, which may incorrectly point to the libGL mesa driver.
  - sudo rm -rf /usr/lib/x86_64-linux-gnu/libGL.so
  5. Link the libGL to the NVIDIA driver
  - sudo ln -s /usr/lib/nvidia-367/libGL.so /usr/lib/x86_64-linux-gnu/libGL.so
  6. Install additional developer dependencies.
  - sudo apt-get install libxinerama-dev
  - sudo apt-get install libxrandr-dev
  - sudo apt-get install libxcursor-dev
  - sudo apt-get install libxi-dev
  - sudo apt-get install libx11-dev

### Install OptiX [optional]
  7. OptiX is distributed as a .sh file, which extracts itself in a desired directory. (Here ~/packages/optix)

### Set up build environment
8. Clone the repository/unpack the source compressed file into a local directory (Here ~/packages/gvdb)
  9. Create a build directory
  ```
  mkdir -P ~/packages/gvdb/build && cd ~/packages/gvdb/build
  ```
### Build cuDPP Library
  10. Create a build directory for cudpp
  ```
  mkdir shared_cudpp && cd shared_cudpp
  ```
11. Configure cudpp and install to local install directory (since the CMakeLists is currently wonky)
  ```
  cmake -DCUDA_SDK_ROOT_DIR=/usr/local/cuda-9.2/samples -DCMAKE_BUILD_TYPE=Release ../../source/shared_cudpp/ -DCMAKE_INSTALL_PREFIX=./install

  make install
  ```
  12. Copy over the include files for cudpp since the CMakeLists isn't correctly configured right now
  ```
  cp -r ../../source/shared_cudpp/include/ ./
  ```

### Build GVDB Library
  13. Create a build directory for gvdb_library
  ```
  cd .. && mkdir gvdb_library && cd gvdb_library
  ```
  14. Invoke cmake to generate config files (Using the appropriate CUDA version) and pointing to above build/install of shared_cudpp, and installing the library in a new root level install directory
  ```
  cmake -DCUDA_SDK_ROOT_DIR=/usr/local/cuda-9.2/samples -DCMAKE_BUILD_TYPE=Release ~/packages/gvdb/source/gvdb_library  -DCUDPP_ROOT_DIR=~/packages/gvdb/build/shared_cudpp -DCMAKE_INSTALL_PREFIX=../../install

  make install
  ```

### Build sample(s)
15. Follow a similar procedure for creating build directories for samples. The appropriate cmake command looks like (an additional command pointing to the extracted optix directory is needed for samples that use optix)
  ```
  cmake -DCUDA_SDK_ROOT_DIR=/usr/local/cuda-9.2/samples -DCMAKE_BUILD_TYPE=Release ~/packages/gvdb/source/gFluidSurface/  -DCUDPP_ROOT_DIR=~/packages/gvdb/build/shared_cudpp -DGVDB_ROOT_DIR=~/packages/gvdb/install -DCMAKE_INSTALL_PREFIX=~/packages/gvdb/install -DOPTIX_ROOT_DIR=~/packages/optix
  ```
### Running the samples
  16. Set LD_LIBRARY_PATH
  ```
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/packages/gvdb/install/bin:~/packages/optix/SDK-precompiled-samples
  ```

  17. Run the sample!
  ```
  cd ~/packages/gvdb/install/bin
  ./gDepthMap
  ```


## License
  ==========================
  BSD 3-clause. Please refer to License.txt


  2017 (c) NVIDIA
