CUDPP documentation                         {#mainpage}
===================

Introduction
============

CUDPP is the CUDA Data Parallel Primitives Library. CUDPP is a
library of data-parallel algorithm primitives such as
parallel-prefix-sum ("scan"), parallel sort and parallel reduction.
Primitives such as these are important building blocks for a wide
variety of data-parallel algorithms, including sorting, stream
compaction, and building data structures such as trees and
summed-area tables.

Overview Presentation
---------------------

A brief set of slides that describe the features, design principles,
applications and impact of CUDPP is available:
[CUDPP Presentation](https://github.com/cudpp/cudpp/blob/master/doc/CUDPP_slides.pdf?raw=true).

Home Page
---------

Homepage for CUDPP: <http://cudpp.github.io/>

Announcements and discussion of CUDPP are hosted on the
[CUDPP Google Group](http://groups.google.com/group/cudpp?hl=en).

Getting Started with CUDPP
--------------------------

You may want to start by browsing the [CUDPP Public Interface](@ref publicInterface).
For information on building CUDPP, see [Building CUDPP](@ref building-cudpp).
See [Overview of CUDPP hash tables](@ref hash_overview) for an overview of CUDPP's hash
table support.

The "apps" subdirectory included with CUDPP has a few source code samples
that use CUDPP:

- [simpleCUDPP](@ref example_simpleCUDPP), a simple example of using cudppScan()
- satGL, an example of using cudppMultiScan() to generate a summed-area table (SAT)
  of a scene rendered in real time.  The SAT is then used to simulate depth of field blur. This example is not currently working due to CUDA graphics interop changes. Volunteers to update it welcome!
- cudpp_testrig, a comprehensive test application for all the functionality of CUDPP
- cudpp_hash_testrig, a comprehensive test application for CUDPP's hash table data structures

We have also provided a code walkthrough of the [simpleCUDPP](@ref example_simpleCUDPP) example.

Getting Help and Reporting Problems
===================================

To get help using CUDPP, please use the [CUDPP Google Group](http://groups.google.com/group/cudpp?hl=en).

To report CUDPP bugs or request features, please file an issue directly using
 [Github](https://github.com/cudpp/cudpp/issues).

Release Notes                               {#release-notes}
=============

For specific release details see the [Change Log](@ref changelog).

Known Issues
------------

For a complete list of issues, see the
[CUDPP issues list](https://github.com/cudpp/cudpp/issues) on Github.

- There is a known issue that the compile time for CUDPP is very long and
  the compiled library file size is very large.  On some systems with < 4GB
  of available memory (or virtual memory: e.g. 32-bit OS), the CUDA compiler
  can run out of memory and compilation can fail. We will be working on these
  issues for future releases. You can reduce compile time by only targetting
  GPU architectures that you plan to run on, using the `CUDPP_GENCODE_*`
  CMake options.
- We have seen "invalid configuration" errors when running
  SM-2.0-compiled suffix array tests on GPUs with SM versions greater
  than 2.0. We see no problems with compiling directly for the GPU's
  native SM version, so the workaround is to compile directly for the
  SM version of your GPU. If you have results or comments on this
  issue, please comment on
  [CUDPP issue 148](https://github.com/cudpp/cudpp/issues/148).

Algorithm Input Size Limitations
--------------------------------

The following maximum size limitations currently apply.  In some
cases this is the theory&mdash;the algorithms may not have been tested
to the maximum size.  Also, for things like 32-bit integer scans,
precision often limits the useful maximum size.

Algorithm            | Maximum Supported Size
---------------------|-----------------------
CUDPP_SCAN           | 67,107,840 elements
CUDPP_SEGMENTED_SCAN | 67,107,840 elements
CUDPP_COMPACT        | 67,107,840 elements
CUDPP_COMPRESS       | 1,048,576 elements
CUDPP_LISTRANK       | NO LIMIT
CUDPP_MTF            | Bounded by GPU memory 
CUDPP_BWT            | 1,048,576 elements
CUDPP_SA             | 0.14 GPU memory
CUDPP_STRINGSORT     | 2,147,450,880 elements
CUDPP_MERGESORT      | 2,147,450,880 elements
CUDPP_MULTISPLIT     | Bounded by GPU memory
CUDPP_REDUCE         | NO LIMIT
CUDPP_RAND           | 33,554,432 elements
CUDPP_SPMVMULT       | 67,107,840 non-zero elements
CUDPP_HASH           | See [Hash Space Limitations](@ref hash_space_limitations)
CUDPP_TRIDIAGONAL    | 65535 systems, 1024 equations per system (Compute capability 2.x), 512 equations per system (Compute capability < 2.0)

Operating System Support and Requirements
=========================================

This release (2.3) has been tested on the following OSes.  For more information, visit our [test results page](https://github.com/cudpp/cudpp/wiki/RegressionStatus2.3).

- Windows 7 (64-bit) (CUDA 6.5)
- Ubuntu Linux (64-bit) (CUDA 6.5)
- Mac OS X 10.12.1 (64-bit) (CUDA 8.0)

We expect CUDPP to build and run correctly on other flavors of Linux and Windows, but only the above are actively tested at this time.  Version 2.3 does not currently support 32-bit operating systems.

Requirements
------------

CUDPP, from this release 2.3 and onwards, now requires a minimum of SM 3.0. CUDPP 2.3 has not been tested with any CUDA version < 6.5.

CUDA
====

CUDPP is implemented in [CUDA C/C++](http://developer.nvidia.com/cuda). It requires the
CUDA Toolkit. Please see the NVIDIA [CUDA](http://developer.nvidia.com/cuda) homepage to
download CUDA as well as the CUDA Programming Guide and CUDA SDK, which includes many
CUDA code examples.

Design Goals
============

Design goals for CUDPP include:

- Performance. We aim to provide best-of-class performance for our primitives.
  We welcome suggestions and contributions that will improve CUDPP performance.
  We also want to provide primitives that can be easily benchmarked, and compared
  against other implementations on GPUs and other processors.
- Modularity. We want our primitives to be easily included in other applications.
  To that end we have made the following design decisions:
  + CUDPP is provided as a library that can link against other applications.
  + CUDPP calls run on the GPU on GPU data. Thus they can be used as standalone
    calls on the GPU (on GPU data initialized by the calling application) and,
    more importantly, as GPU components in larger CPU/GPU applications.
- CUDPP is implemented as 4 layers:
  + The [Public Interface](@ref publicInterface) is the external library interface,
    which is the intended entry point for most applications. The public interface
    calls into the [Application-Level API](@ref cudpp_app).
  + The [Application-Level API](@ref cudpp_app) comprises functions callable from CPU code.
    These functions execute code jointly on the CPU (host) and the GPU by calling
    into the [Kernel-Level API](@ref cudpp_kernel) below them.
  + The [Kernel-Level API](@ref cudpp_kernel) comprises functionsthat run entirely on
    the GPU across an entire grid of thread blocks. These functions may call into
    the [CTA-Level API](@ref cudpp_cta) below them.
  + The [CTA-Level API](@ref cudpp_cta) comprises functions that run entirely on the
    GPU within a single Cooperative Thread Array (CTA, aka a CUDA thread block).
    These are low-level functions that implement core data-parallel algorithms,
    typically by processing data within shared (CUDA `__shared__`) memory.

Programmers may use any of the lower three CUDPP layers in their own programs by
building the source directly into their application. However, the typical usage
of CUDPP is to link to the library and invoke functions in the CUDPP
[Public Interface](@ref publicInterface), as in the [simpleCUDPP](@ref example_simpleCUDPP),
satGL, cudpp_testrig, and cudpp_hash_testrig application examples included in the
CUDPP distribution.

Use Cases
---------

We expect the normal use of CUDPP will be in one of two ways:

- Linking the CUDPP library against another application.
- Running the "test" applications, cudpp_testrig and cudpp_hash_testrig, that
  exercise CUDPP functionality.

References {#references}
==========

The following publications describe work incorporated in CUDPP.

- Mark Harris, Shubhabrata Sengupta, and John D. Owens. "Parallel Prefix Sum (Scan) with CUDA". In Hubert Nguyen, editor, <i>GPU Gems 3</i>, chapter 39, pages 851&ndash;876. Addison Wesley, August 2007. http://www.idav.ucdavis.edu/publications/print_pub?pub_id=916
- Shubhabrata Sengupta, Mark Harris, Yao Zhang, and John D. Owens. "Scan Primitives for GPU Computing". In <i>Graphics Hardware 2007</i>, pages 97&ndash;106, August 2007. http://www.idav.ucdavis.edu/publications/print_pub?pub_id=915
- Nadathur Satish, Mark Harris, and Michael Garland. "Designing Efficient Sorting Algorithms for Manycore GPUs". In <i>Proceedings of the 23rd IEEE International Parallel & Distributed Processing Symposium</i>, May 2009. http://mgarland.org/papers.html#gpusort
- Stanley Tzeng, Li-Yi Wei. "Parallel White Noise Generation on a GPU via Cryptographic Hash". In <i>Proceedings of the 2008 Symposium on Interactive 3D Graphics and Games</i>, pages 79&ndash;87, February 2008. http://research.microsoft.com/apps/pubs/default.aspx?id=70502
- Yao Zhang, Jonathan Cohen, and John D. Owens. Fast Tridiagonal Solvers on the GPU. In <i>Proceedings of the 15th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP 2010)</i>, pages 127&ndash;136, January 2010. http://www.cs.ucdavis.edu/publications/print_pub?pub_id=978
- Yao Zhang, Jonathan Cohen, Andrew A. Davidson, and John D. Owens. A Hybrid Method for Solving Tridiagonal Systems on the GPU. In Wen-mei W. Hwu, editor, <i>GPU Computing Gems</i>. Morgan Kaufmann. July 2011.
- Shubhabrata Sengupta, Mark Harris, Michael Garland, and John D. Owens. "Efficient Parallel Scan Algorithms for many-core GPUs". In Jakub Kurzak, David A. Bader, and Jack Dongarra, editors, <i>Scientific Computing with Multicore and Accelerators</i>, Chapman & Hall/CRC Computational Science, chapter 19, pages 413&ndash;442. Taylor & Francis, January 2011. http://www.idav.ucdavis.edu/publications/print_pub?pub_id=1041
- Dan A. Alcantara, Andrei Sharf, Fatemeh Abbasinejad, Shubhabrata Sengupta, Michael Mitzenmacher, John D. Owens, and Nina Amenta. Real-Time Parallel Hashing on the GPU. ACM Transactions on Graphics, 28(5):154:1–154:9, December 2009. http://www.idav.ucdavis.edu/publications/print_pub?pub_id=973
- Dan A. Alcantara, Vasily Volkov, Shubhabrata Sengupta, Michael Mitzenmacher, John D. Owens, and Nina Amenta. Building an Efficient Hash Table on the GPU. In Wen-mei W. Hwu, editor, GPU Computing Gems, volume 2, chapter 1. Morgan Kaufmann, August 2011.
- Ritesh A. Patel, Yao Zhang, Jason Mak, Andrew Davidson, John D. Owens. "Parallel Lossless Data Compression on the GPU". In <i>Proceedings of Innovative Parallel Computing (InPar '12)</i>, May 2012. http://idav.ucdavis.edu/publications/print_pub?pub_id=1087
- Andrew Davidson, David Tarjan, Michael Garland, and John D. Owens. Efficient Parallel Merge Sort for Fixed and Variable Length Keys. <i>In Proceedings of Innovative Parallel Computing (InPar '12)</i>, May 2012. http://www.idav.ucdavis.edu/publications/print_pub?pub_id=1085
- Saman Ashkiani, Andrew Davidson, Ulrich Meyer, and John D. Owens. GPU Multisplit. <i>In Proceedings of the 21st ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '16)</i>, March 2016, http://escholarship.org/uc/item/346486j8 

</pre>
Many researchers are using CUDPP in their work, and there are many publications that
have used it ([references](@ref cudpp_refs)). If your work uses CUDPP, please let us know
by sending us a reference (preferably in BibTeX format) to your work.

Citing CUDPP
============

If you make use of CUDPP primitives in your work and want to cite CUDPP (thanks!),
we would prefer for you to cite the appropriate papers above, since they form the core
of CUDPP. To be more specific, the GPU Gems paper (Harris et al.) describes
(unsegmented) scan, multi-scan for summed-area tables, and stream compaction.
The Sengupta et al. book chapter describes the current scan and segmented scan
algorithms used in the library, and the Sengupta et al. Graphics Hardware paper
describes an earlier implementation of segmented scan, quicksort, and sparse
matrix-vector multiply. The IPDPS paper (Satish et al.) describes the radix sort
used in CUDPP (prior to CUDPP 2.0. Later releases use Thrust::sort), and the I3D
paper (Tzeng and Wei) describes the random number generation algorithm. The two
Alcantara papers describe the hash algorithms. The two Zhang papers describe the
tridiagonal solvers.

Credits
=======

CUDPP Developers
----------------

- [Mark Harris](http://www.markmark.net), NVIDIA Corporation
- [John D. Owens](http://www.ece.ucdavis.edu/~jowens/), University of California, Davis
- [Shubho Sengupta](http://graphics.cs.ucdavis.edu/~shubho/), University of California, Davis
- [Stanley Tzeng](http://csiflabs.cs.ucdavis.edu/~stzeng/), University of California, Davis
- [Yao Zhang](http://www.alcf.anl.gov/~yaozhang/), University of California, Davis
- [Andrew Davidson](http://www.ece.ucdavis.edu/~aaldavid/), University of California, Davis
- [Ritesh Patel](http://www.ece.ucdavis.edu/~ritesh88/), University of California, Davis
- [Leyuan Wang](http://www.ece.ucdavis.edu/~laurawly/), University of California, Davis
- [Saman Ashkiani](http://www.ece.ucdavis.edu/~ashkiani/), University of California, Davis

Other CUDPP Contributors
------------------------

- [Jason Mak](http://idav.ucdavis.edu/~jwmak/), University of California, Davis [Release Manager]
- [Anjul Patney](http://idav.ucdavis.edu/~anjul/), University of California, Davis [general help]
- [Edmund Yan](http://csiflabs.cs.ucdavis.edu/~eyan/), University of California, Davis [Release Manager]
- [Dan Alcantara](http://idav.ucdavis.edu/~dfalcant/research.php), University of California, Davis [hash tables]
- [Nadatur Satish](http://pcl.intel-research.net/people/satish.htm), University of California, Berkeley [(old)radix sort]

Acknowledgments
---------------

Thanks to Jim Ahrens, Timo Aila, Nathan Bell, Ian Buck, Guy Blelloch,
Jeff Bolz, Michael Garland, Jeff Inman, Eric Lengyel, Samuli Laine,
David Luebke, Pat McCormick, Duane Merrill, and Richard Vuduc for their
contributions during the development of this library.

CUDPP Developers from UC Davis thank their funding agencies:

- National Science Foundation (grants CCF-0541448, IIS-0964357, and particularly OCI-1032859)
- Department of Energy Early Career Principal Investigator Award DE-FG02-04ER25609
- SciDAC Institute for Ultrascale Visualization (http://www.iusv.org/)
- Los Alamos National Laboratory
- Generous hardware donations from NVIDIA

CUDPP Copyright and Software License
====================================

CUDPP is copyright The Regents of the University of California, Davis campus
and NVIDIA Corporation.  The library, examples, and all source code are
released under the BSD license, designed to encourage reuse of this software
in other projects, both commercial and non-commercial.  For details, please
see the [license](@ref license) page.

Non source-code content (such as documentation, web pages, etc.) from CUDPP
is distributed under a [Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)](http://creativecommons.org/licenses/by-sa/3.0/) license.

Note that prior to release 1.1 of CUDPP, the license used was a modified
BSD license.  With release 1.1, this license was replaced with the pure BSD
license to facilitate the use of open source hosting of the code.

CUDPP also includes the [Mersenne twister code](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html) of [Makoto Matsumoto](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/eindex.html),
also licensed under BSD.

CUDPP also calls functions in the [Thrust](http://thrust.github.io) template library,
which is included with the CUDA Toolkit and licensed under the Apache 2.0 open source
license.

CUDPP also includes a modified version of FindGLEW.cmake from
[nvidia-texture-tools](http://code.google.com/p/nvidia-texture-tools/),
licensed under the [MIT license](http://www.opensource.org/licenses/mit-license.php).
