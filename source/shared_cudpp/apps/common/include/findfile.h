// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//  $Source$
//  $Revision: 3632 $
//  $Date: 2007-08-26 06:15:39 +0100 (Sun, 26 Aug 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * findFile.h
 * 
 * @brief functions to find files and directories
 */

#ifndef _TOOLS_H_
#define _TOOLS_H_

namespace cudpp_app {

    //Dir/File searching functions
    //wrapper functions for the whole routine
    bool findDir(const char * startDir, const char * dirName, char * outputPath);
    bool findFile(const char * startDir, const char * dirName, char * outputPath);

}

#ifdef CUDPP_APP_COMMON_IMPL
#include "findfile.inl"
#endif

#endif  //#ifndef _TOOLS_H_

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
