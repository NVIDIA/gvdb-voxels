// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * arraycompare.h
 * 
 * @brief Templatization of array comparisons 
 * 
 */

#ifndef _COMPARE_ARRAYS_H_
#define _COMPARE_ARRAYS_H_

#include <assert.h>
#include <fstream>
#include <iostream>

namespace cudpp_app {
    
    template<typename T>
    bool compareArrays( const T* reference, const T* data, unsigned int len, float epsilon = 0)
    {
        assert( epsilon >= 0);

        bool result = true;
        unsigned int error_count = 0;

        for( unsigned int i = 0; i < len; ++i) {

            T diff = reference[i] - data[i];
            bool comp = (diff <= epsilon) && (diff >= -epsilon);
            result &= comp;

            error_count += !comp;

    #ifdef _DEBUG
            if( ! comp) 
            {
                std::cerr << "ERROR, i = " << i << ",\t " 
                    << reference[i] << " / "
                    << data[i] 
                    << " (reference / data)\n";
            }
    #endif
        }

        return (result) ? true : false;
    }

    template <typename T>
    bool writeArrayToFile(const char* filename, T* data, unsigned int len, float epsilon)
    {
        assert( NULL != filename);
        assert( NULL != data);

        // open file for writing
        std::fstream fh( filename, std::fstream::out);
        // check if filestream is valid
        if( ! fh.good()) 
        {
            std::cerr << "writeArrayToFile() : Opening file failed." << std::endl;
            return false;
        }

        // first write epsilon
        fh << "# " << epsilon << "\n";

        // write data
        for( unsigned int i = 0; (i < len) && (fh.good()); ++i) 
        {
            fh << data[i] << ' ';
        }

        // Check if writing succeeded
        if( ! fh.good()) 
        {
            std::cerr << "writeArrayToFile() : Writing file failed." << std::endl;
            return false;
        }

        // file ends with nl
        fh << std::endl;

        return true;
    }
}

#endif //_COMPARE_ARRAYS_H_

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
