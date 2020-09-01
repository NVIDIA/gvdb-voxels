NVIDIA® GVDB VOXELS
Release 1.1.1

## OVERVIEW
NVIDIA® GVDB Voxels is a new library and SDK for simulation, compute, and rendering of sparse volumetric data. Details on the GVDB technology can be found at [http://developer.nvidia.com/gvdb](http://developer.nvidia.com/gvdb).

## RELEASE NOTES

1/29/2020, GVDB Voxels 1.1.1

* Significantly streamlined CMake build system
* gImportVDB Linux support
* Uses surface and texture objects instead of surface and texture references
* Remove dependency on CUDPP, now using Thrust for radix sorting and reduction
* Watertight voxelization robustness improvements, using Woop, Benthin, and Wald's watertight rasterization technique and D3D10-style top-left edge rules
* Remove limitation of 65,535 bricks in UpdateApron and UpdateApronFaces
* Voxel Size has been removed (i.e. GVDB's world space is now equal to its index space; please use SetTransform to apply your own arbitrary affine transformations)
* Many bug and formatting fixes (including [public contributions](https://github.com/NVIDIA/gvdb-voxels/graphs/contributors))

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
Created by Rama Hoetzlein, 2017

- First public release
- Open source (BSD 3-clause)
- Samples: g3DPrint, gFluidSurface, gInteractiveGL, gInteractiveOptix,
  gJetsonTX, gRenderKernel, gRenderToFile, gSprayDeposit
- Builds on Windows and Linux
- Runs on Quadro, GeForce, JetsonTX1/2, and Tegra/GRID


## REQUIREMENTS
- NVIDIA Kepler generation or later GPU
- CUDA Toolkit 6.5 or higher (Toolkit 10.2 recommended)
- CMake 3.10 or later

GVDB is released as a library with samples. The library and each sample can be built separately, using CMake.

## WHAT'S IN THE PACKAGE?

  - The GVDB Voxels Library
  - Code Samples
      - **g3DPrint** - Demonstrates generating cross-section slices for 3D printing from a polygonal model.
      - **gDepthMap** - Renders a polygonal mesh and an emissive volume at the same time using OpenGL and CUDA.
      - **gFluidSurface** - A dynamic fluid simulation with ray traced surface rendering using GVDB and OptiX.
      - **gImportVDB** - Loads and ray traces a volume stored in OpenVDB's .vdb format.
      - **gInteractiveGL** - Interactive rendering of a volume using GVDB and OpenGL.
      - **gInteractiveOptix** - Interactive ray tracing of a volume and a mesh, with light bouncing between the two.
      - **gNanoVDB** - Exports a GVDB volume to a NanoVDB volume, then renders it using NanoVDB.
      - **gPointFusion** - Fuses points from a moving camera into a full 3D volume.
      - **gRenderKernel** - Rendering without OpenGL using a custom GVDB kernel.
      - **gRenderToFile** - Rendering a semitransparent object to a file without OpenGL using GVDB.
      - **gResample** - Imaging of a sparse volume generated from dense data.
      - **gSprayDeposit** - Demonstrates simulated spray deposition onto a 3D park.
      - **gJetsonTX** - Simple 3D printing driver for the Jetson TX1/2 with volume slicing on a Tegra chip.
  - The GVDB VBX File Specification
  - GVDB Sample Descriptions
  - The GVDB Programming Guide
## SAMPLE USAGE
All interactive samples use the following user interface:

* Camera rotation -> move mouse
* Change orientation -> left mouse click

* Zoom -> right mouse click

* Panning -> hold middle button

A few samples have on-screen GUIs with features that can be toggled by clicking on them.

## QUICK BUILD PROCESS (Windows and Linux)

### Install dependencies
  1. Install [CMake 3.10](https://cmake.org/download/) or later.
  2. Install [CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-downloads) or later.

### To build the samples and the library at the same time:

3. In CMake, configure and generate the build system using gvdb-voxels/CMakeLists.txt, then build the `BUILD_ALL` target in your IDE (such as Visual Studio).

That's it!

You can also build a specific sample or the GVDB library this way by building its target in this project. Additionally, you can collect a build of GVDB and its samples into a redistributable package by building the `INSTALL` target.

(Wondering what the `GVDB_BUILD_OPTIX_SAMPLES`, `GVDB_BUILD_OPENVDB`, `GVDB_BUILD_OPENVDB_SAMPLES`, `GVDB_BUILD_NANOVDB`, and `GVDB_BUILD_NANOVDB` checkboxes in the CMake GUI do? See "To build the OptiX samples", "To build GVDB with OpenVDB", and "To build the NanoVDB sample" below.)

### To build the GVDB Library by itself:

3. In CMake, configure and generate the build system using `gvdb-voxels/source/gvdb_library/CMakeLists.txt`, then build the `gvdb` target.

As above, you can create a redistributable build of the NVIDIA GVDB Voxels library by building the `INSTALL` target.

### To build a sample by itself:

3. In CMake, configure and generate the build system using `gvdb-voxels/source/[your sample name here]/CMakeLists.txt`, then build your sample's target.

That's it! In Visual Studio, you can also run a sample by right-clicking it inside Visual Studio, selecting "Set as StartUp Project", and then pressing F5 or clicking the green triangle in the toolbar.

For some samples, you'll see targets named things like `gSample`, `gSampleApp`, and `gSamplePTX`. In this case, you'll want to build and run `gSample`; the other two targets compile the application and its PTX files, while the `gSample` target collects everything together.

Building a sample will also automatically build GVDB, so you no longer need to build and install GVDB before building a sample.

### To build the OptiX samples:

1. Install OptiX 6.5 from https://developer.nvidia.com/designworks/optix/download.
2. On **Windows**, check `GVDB_BUILD_OPTIX_SAMPLES` in the CMake GUI, or add `-DGVDB_BUILD_OPTIX_SAMPLES=ON` to the CMake command line.
3. On **Linux**, check `GVDB_BUILD_OPTIX_SAMPLES` in the CMake GUI, and also add an entry, `OPTIX_ROOT_DIR`, pointing to the path to the OptiX SDK (the folder containing OptiX's `lib64` directory). Or if you're using the CMake command line, add `-DGVDB_BUILD_OPTIX_SAMPLES=ON -DOPTIX_ROOT_DIR=<path to OptiX SDK>`, replacing `<path to OptiX SDK>` with the correct path.
4. Finally, generate and build the CMake project.

### To build GVDB with OpenVDB:

**Windows:**

1. Install OpenVDB: On Windows, one of the easiest ways to install OpenVDB is to use Microsoft's [vcpkg](https://github.com/microsoft/vcpkg); install `vcpkg`, then run `vcpkg install openvdb[tools]:x64-windows`. Make sure `vcpkg` is using the same compiler you'll use to compile GVDB!
2. Work around a temporary issue in vcpkg: If you plan to build GVDB in release mode, go to your `vcpkg/installed/x64-windows-debug/bin` folder and copy `openvdb_d.dll` to `openvdb.dll`. This works around an issue where a build system copies debug-mode `openvdb.lib` and `openvdb.dll` to `openvdb_d.lib` and `openvdb_d.dll` respectively, but doesn't update the DLL reference in `openvdb_d.lib`.
3. Configure CMake:
   1. If you're using the **CMake GUI**, delete the cache, then click the "Configure" button, specify your generator and platform, check "Specify toolchain file for cross-compiling", and click "Next". Then specify the path to `vcpkg/scripts/buildsystems/vcpkg.cmake`, and click Finish. Then check `GVDB_BUILD_OPENVDB` (and if you'd like to build the gImportVDB sample as well, check `GVDB_BUILD_OPENVDB_SAMPLES`) and click "Configure" again.
   2. If you're using the **CMake command line**, you can also do this by specifying `-DCMAKE_TOOLCHAIN_FILE=<path to vcpkg.cmake> -DGVDB_BUILD_OPENVDB=ON -DGVDB_BUILD_OPENVDB_SAMPLES=ON`.
   3. Alternatively, if you're not using `vcpkg`, you can also specify `GVDB_OPENVDB_INCLUDE_DIR`, `GVDB_OPENVDB_LIB_RELEASE_DIR`, `GVDB_OPENVDB_LIB_DEBUG_DIR`, and `GVDB_OPENVDB_DLL_RELEASE_DIR` and copy in OpenVDB's DLLs using any method - see `gvdb_library/CMakeLists.txt` for more information.
4. Finally, generate and build the CMake project. Now you can run GVDB with OpenVDB!

**Linux:**

1. Install OpenVDB 6.1+: On Linux, we recommend building OpenVDB from source using the instructions on [OpenVDB's Developer Quick Start](https://github.com/AcademySoftwareFoundation/openvdb/#linux), unless OpenVDB 6.1+ is available through your distro's package manager (6.1 introduced a new CMake build system in OpenVDB that we rely upon). Note that you may have to add `-DCMAKE_NO_SYSTEM_FROM_IMPORTED:BOOL=TRUE` if you run into OpenVDB [Issue 144](https://github.com/AcademySoftwareFoundation/openvdb/issues/144#issuecomment-508984426).
2. Configure CMake:
   1. If you're using the **CMake GUI**, check `GVDB_BUILD_OPENVDB` and `GVDB_BUILD_OPENVDB_SAMPLES` and click "Configure" again.
   2. If you're using the **CMake command line**, you can also do this by specifying `-DGVDB_BUILD_OPENVDB=ON -DGVDB_BUILD_OPENVDB_SAMPLES=ON`.
3. Finally, generate and build the CMake project. Now you can run GVDB with OpenVDB!

### To build the NanoVDB sample:

1. Download NanoVDB from the OpenVDB repository at https://github.com/AcademySoftwareFoundation/openvdb/tree/feature/nanovdb/nanovdb. Since NanoVDB is a header-only library, there's no need to build OpenVDB.
2. In CMake, set `GVDB_NANOVDB_INCLUDE_DIR` to the path to NanoVDB (this folder contains a `nanovdb` folder which contains `NanoVDB.h`), set `GVDB_BUILD_NANOVDB` to `ON`, and set `GVDB_BUILD_NANOVDB_SAMPLES` to `ON`.
3. Finally, generate and build the CMake project. Now you can run the `gNanoVDB` sample!


## License

==========================

BSD 3-clause. Please refer to [LICENSE.txt](https://github.com/NVIDIA/gvdb-voxels/blob/master/LICENSE.txt).

© 2020 NVIDIA
