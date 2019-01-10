// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * ppm.h
 * 
 * @brief functions to find files and directories
 */

#ifndef _PPM_H_
#define _PPM_H_

namespace cudpp_app {

    //////////////////////////////////////////////////////////////////////////////
    //! Load PGM or PPM file
    //! @note if data == NULL then the necessary memory is allocated in the 
    //!       function and w and h are initialized to the size of the image
    //! @return true if the file loading succeeded, otherwise false
    //! @param file        name of the file to load
    //! @param data        handle to the memory for the image file data
    //! @param w        width of the image
    //! @param h        height of the image
    //! @param channels number of channels in image
    //////////////////////////////////////////////////////////////////////////////
    bool loadPPM( const char* file, unsigned char** data, 
                  unsigned int *w, unsigned int *h, unsigned int *channels ); 

}

#ifdef CUDPP_APP_COMMON_IMPL
#include "ppm.inl"
#endif

#endif // _PPM_H_