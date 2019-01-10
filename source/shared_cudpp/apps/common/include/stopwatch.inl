// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: $
// $Date: $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * stopwatch.cpp
 *
 * @brief Timer class
 */

#ifdef CUDPP_APP_COMMON_IMPL

namespace cudpp_app {

#ifdef WIN32
    //! tick frequency
    /*static*/ double  StopWatch::freq;

    //! flag if the frequency has been set
    /*static*/  bool   StopWatch::freq_set;
#endif

    ////////////////////////////////////////////////////////////////////////////////
    //! Constructor, default
    ////////////////////////////////////////////////////////////////////////////////
    StopWatch::StopWatch() :
        start_time(),
    #ifdef WIN32
        end_time(),
    #endif
        diff_time( 0.0),
        total_time(0.0),
        running( false),
        clock_sessions(0)
    {
    #ifdef WIN32
        if( ! freq_set) 
        {
            // helper variable
            LARGE_INTEGER temp;

            // get the tick frequency from the OS
            QueryPerformanceFrequency((LARGE_INTEGER*) &temp);

            // convert to type in which it is needed
            freq = ((double) temp.QuadPart) / 1000.0;

            // rememeber query
            freq_set = true;
        }
    #endif
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Destructor
    ////////////////////////////////////////////////////////////////////////////////
    StopWatch::~StopWatch() { }

}

#endif