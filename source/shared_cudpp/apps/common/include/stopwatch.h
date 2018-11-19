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
 * stopwatch.h
 *
 * @brief Timer class
 */

#ifndef _STOPWATCH_H_
#define _STOPWATCH_H_

#ifdef WIN32
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max
#else
#include <ctime>
#include <sys/time.h>
#endif

namespace cudpp_app {
    
    class StopWatch
    {
    public:
        //! Constructor, default
        StopWatch();

        // Destructor
        ~StopWatch();

        //! Start time measurement
        inline void start();

        //! Stop time measurement
        inline void stop();

        //! Reset time counters to zero
        inline void reset();

        //! Time in msec. after start. If the stop watch is still running (i.e. there
        //! was no call to stop()) then the elapsed time is returned, otherwise the
        //! time between the last start() and stop call is returned
        inline const float getTime() const;

        //! Mean time to date based on the number of times the stopwatch has been 
        //! _stopped_ (ie finished sessions) and the current total time
        inline const float getAverageTime() const;
    
    private:     // member variables
    #ifdef WIN32
        //! Start of measurement
        LARGE_INTEGER  start_time;
        //! End of measurement
        LARGE_INTEGER  end_time;

        //! tick frequency
        static double  freq;

        //! flag if the frequency has been set
        static  bool  freq_set;
    #else
        //! Get difference between start time and current time
        inline float getDiffTime() const;

        //! Start of measurement
        struct timeval  start_time;
    #endif

        //! Time difference between the last start and stop
        float  diff_time;

        //! TOTAL time difference between starts and stops
        float  total_time;

        //! flag if the stop watch is running
        bool running;

        //! Number of times clock has been started
        //! and stopped to allow averaging
        int clock_sessions;
    };


    #ifdef WIN32
    ////////////////////////////////////////////////////////////////////////////////
    //! Start time measurement
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatch::start() 
    {
        QueryPerformanceCounter((LARGE_INTEGER*) &start_time);
        running = true;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Stop time measurement and increment add to the current diff_time summation
    //! variable. Also increment the number of times this clock has been run.
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatch::stop() 
    {
        QueryPerformanceCounter((LARGE_INTEGER*) &end_time);
        diff_time = (float) 
            (((double) end_time.QuadPart - (double) start_time.QuadPart) / freq);

        total_time += diff_time;
        clock_sessions++;
        running = false;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Reset the timer to 0. Does not change the timer running state but does 
    //! recapture this point in time as the current start time if it is running.
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatch::reset() 
    {
        diff_time = 0;
        total_time = 0;
        clock_sessions = 0;
        if( running )
            QueryPerformanceCounter((LARGE_INTEGER*) &start_time);
    }


    ////////////////////////////////////////////////////////////////////////////////
    //! Time in msec. after start. If the stop watch is still running (i.e. there
    //! was no call to stop()) then the elapsed time is returned added to the 
    //! current diff_time sum, otherwise the current summed time difference alone
    //! is returned.
    ////////////////////////////////////////////////////////////////////////////////
    inline const float 
    StopWatch::getTime() const 
    {
        // Return the TOTAL time to date
        float retval = total_time;
        if(running) 
        {
            LARGE_INTEGER temp;
            QueryPerformanceCounter((LARGE_INTEGER*) &temp);
            retval += (float) 
                (((double) (temp.QuadPart - start_time.QuadPart)) / freq);
        }

        return retval;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Time in msec. for a single run based on the total number of COMPLETED runs
    //! and the total time.
    ////////////////////////////////////////////////////////////////////////////////
    inline const float 
    StopWatch::getAverageTime() const
    {
    	return (clock_sessions > 0) ? (total_time/clock_sessions) : 0.0f;
    }
    #else
    ////////////////////////////////////////////////////////////////////////////////
    //! Start time measurement
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatch::start() {

      gettimeofday( &start_time, 0);
      running = true;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Stop time measurement and increment add to the current diff_time summation
    //! variable. Also increment the number of times this clock has been run.
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatch::stop() {

      diff_time = getDiffTime();
      total_time += diff_time;
      running = false;
      clock_sessions++;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Reset the timer to 0. Does not change the timer running state but does 
    //! recapture this point in time as the current start time if it is running.
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatch::reset() 
    {
      diff_time = 0;
      total_time = 0;
      clock_sessions = 0;
      if( running )
        gettimeofday( &start_time, 0);
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Time in msec. after start. If the stop watch is still running (i.e. there
    //! was no call to stop()) then the elapsed time is returned added to the 
    //! current diff_time sum, otherwise the current summed time difference alone
    //! is returned.
    ////////////////////////////////////////////////////////////////////////////////
    inline const float 
    StopWatch::getTime() const 
    {
        // Return the TOTAL time to date
        float retval = total_time;
        if( running) {

            retval += getDiffTime();
        }

        return retval;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Time in msec. for a single run based on the total number of COMPLETED runs
    //! and the total time.
    ////////////////////////////////////////////////////////////////////////////////
    inline const float 
    StopWatch::getAverageTime() const
    {
        return (clock_sessions > 0) ? (total_time/clock_sessions) : 0.0f;
    }



    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    inline float
    StopWatch::getDiffTime() const 
    {
      struct timeval t_time;
      gettimeofday( &t_time, 0);

      // time difference in milli-seconds
      return  (float) (1000.0 * ( t_time.tv_sec - start_time.tv_sec) 
                    + (0.001 * (t_time.tv_usec - start_time.tv_usec)) );
    }
    #endif

}

#ifdef CUDPP_APP_COMMON_IMPL
#include "stopwatch.inl"
#endif

#endif // _STOPWATCH_H_

