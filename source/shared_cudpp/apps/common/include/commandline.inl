// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * commandline.inl
 * 
 * @brief Command line argument parsing
 * 
 */
#ifdef CUDPP_APP_COMMON_IMPL

namespace cudpp_app {
    
    bool checkCommandLineFlag(int argc, const char** argv, const char* name)
    {
        bool val = false;
        commandLineArg(val, argc, argv, name);
        return val;
    }

}

#endif