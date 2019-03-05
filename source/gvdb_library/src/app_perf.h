//--------------------------------------------------------------------------------
//
// File:   app_perf.h
// 
// Copyright (c) 2015, NVIDIA. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//----------------------------------------------------------------------------------
// Created: Rama Hoetzlein, rhoetzlein@nvidia.com, 2013
//
// This lightweight performance class provides additional
// features for CPU and GPU profiling:
// 1. By default, markers are disabled if nvToolsExt32_1.dll 
//    is not found in the working path. Useful when shipping a product.
// 2. Providing nvToolsExt32_1.dll automatically enables CPU and GPU markers.
// 3. If nvToolsExt_32.dll is not present, you can still 
//    enable CPU only markers by calling PERF_INIT(false);  // false = don't require dll
// 4. Instrument code with single PERF_PUSH, PERF_POP markers 
//    for both CPU and GPU output.
// 5. Perform additional printing along with markers using PERF_PRINTF
// 6. Output CPU markers to log file by specifing filename to PERF_SET
// 7. Markers can be nested, with range output for both CPU and GPU
// 8. Only app_perf.h and app_perf.cpp are needed. No need to link with nvToolsExt.h.
// 9. CPU and GPU can be enabled selectively in different parts of the app.
//    Call PERF_SET( CPUon?, CPUlevel, GPUon?, LogFilename ) at any time.
// 10. CPU times reported in milliseconds, with sub-microsecond accuracy.
// 11. CPU Level specifies maximum printf level for markers. Useful when
//     your markers are inside an inner loop. You can keep them in code, but hide their output.
// 12. GPU markers use NVIDIA's Perfmarkers for viewing in NVIDIA NSIGHT
// 

#ifndef APP_PERF
  #define APP_PERF

	#include <stdio.h>
	#include "gvdb_types.h"

	//----------------- PERFORMANCE MARKERS

	#if !defined(_WIN32)
	#	define __stdcall /* no __stdcall stuff on non-Windows platforms. */
	#endif

	#ifdef USE_NVTX
		typedef int ( __stdcall *nvtxRangePushFunc)(const char* msg);
		typedef int ( __stdcall *nvtxRangePopFunc)(void);
	#endif

	extern "C" GVDB_API void PERF_PUSH ( const char* msg );
	extern "C" GVDB_API float PERF_POP ();
	extern "C" GVDB_API void PERF_START ();
	extern "C" GVDB_API float PERF_STOP ();
	extern "C" GVDB_API void PERF_INIT ( int buildbits, bool cpu, bool gpu, bool cons, int lev, const char* fname );
	extern "C" GVDB_API void PERF_SET ( bool cons, int lev );
	extern "C" GVDB_API void PERF_PRINTF ( char* format, ... );


	// Time Class
	// Copyright Rama Hoetzlein (C) 2007
	// This code is released under BSD 3-Clause License. (https://opensource.org/licenses/BSD-3-Clause)
	//
	// Overview:
	//  There is a need in many systems to represent both very small (nanoseconds) and 
	//  very large (millenia) timescales accurately. Modified Julian Date accurate represents 
	//  individual days over +/- about 30,000 yrs. However, MJD represents fractions of a day
	//  as a floating point fraction. This is inaccurate for any timing-critical applications.	
	//  The Time class here uses an 8-byte (64 bit) integer called SJT, Scaled Julian Time.
	//      SJT = MJD * DAY_SCALAR + UT (nanoseconds).
	//  SJT is the Modified Julian Date scaled by a integer factor, and added to Universal Time
	//  represented in nanoseconds.
	//
	// Features:
	// - Accurately represents individual nanoseconds over +/- 30,000 yrs.
	// - Correct rollover of tiny time scales on month, day, year boundaries.
	//     e.g. Set date/time to 11:59:59.9999, on Feb 28th, 
	// - Accurately gives day of the week for any date
	// - Accurately compares two dates (days elapsed) even across leap-years.
	// - Adjust sec/nsec independently from month/day/year (work at scale you desire)	
	//
	// Implementation Notes:
	// JD = Julian Day is the number of days elapsed since Jan 1, 4713 BC in the proleptic Julian calendar.
	//     http://en.wikipedia.org/wiki/Julian_day
	// MJD = Modified Julian Date. Most modern dates, after 19th c., have Julian Date which are greater
	//       than 2400000.5. MJD is an offset. MJD = JD - 2400000.5
	//       It shifts the epoch date (start date) to Nov 17, 1858.
	// UT = Universal Time. This is the time of day in hours as measured from Greenwich England.
	//       For non-astronomic uses, this is: UT = Local Time + Time Zone.
	// SJT = Scaled Julian Time = MJD * DAY_SCALAR + UT (in nanoseconds).
	//
	// Julian Dates (and their MJD and SJT equivalents)
	// ------------
	// Jan 1,  4713 BC = JD 0			= MJD -2400000	= SJT 
	// Jan 1,  1500 AD = JD 2268933.5	= MJD -131067	= SJT 
	// Nov 16, 1858 AD = JD 2400000.5	= MJD 0			= SJT 0
	// Jan 1,  1960 AD = JD 2436935.5	= MJD 36935		= SJT 3,191,184,000,000
	// Jan 1,  2005 AD = JD 2453372.5	= MJD 53372		= SJT 4,611,340,800,000
	// Jan 1,  2100 AD = JD 2488070.5	= MJD 88070		= SJT 7,609,248,000,000
	//
	//
	// 
	// 32/64-Bit Integer Ranges
	//    32-bit Integer Min:              –2,147,483,648   ( 4 bytes )
	//    32-bit Integer Max:               2,147,483,647 
	//    SJT 2005:                     4,611,340,800,000
	//    64-bit Integer Min:  –9,223,372,036,854,775,808 
	//    64-bit Integer Max:   9,223,372,036,854,775,807	( 8 bytes )
	//
	// SJT Range
	// ---------
	//   * USING DAY_SCALAR = 86,400,000 (millisec accuracy)
	//   SJT Range = (+/-9,223,372,036,854,775,807 SJT / 86,400,000 DAY_SCALAR)
	//   SJT Range (in Julian Days) = +2400000.5 + (+/-106,751,991,167 MJD) 	
	//   SJT Range (in Julian Days) = +/- 292278883 years, with 1 millisecond accuracy.
	//
	//   * USING DAY_SCALAR = 86,400,000,000,000 (nanosec accuracy)	
	//   SJT Range = (+/-9,223,372,036,854,775,807 SJT / 86,400,000,000,000 DAY_SCALAR)	
	//   SJT Range (in Julian Days) = +2400000.5 + (+/-106,751 MJD)	
	//   SJT Range (in Julian Days) = 1566 AD to 2151 AD, with 1 nanosecond accuracy.

	#include <string>	
	#ifdef _MSC_VER
		#include <windows.h>
		typedef __int64		mstime;
		typedef __int64		sjtime;
	#endif
	#ifdef __linux__
		#include <linux/types.h>
		typedef __s64       mstime;
		typedef __s64       sjtime;
	#endif

	#ifdef __CYGWIN__	
		#include <largeint.h>	
		typedef __int64       mstime;
		typedef __int64       sjtime;
	#endif	

	#include <string>	
	#ifdef _MSC_VER
		#define MSEC_SCALAR		1000000i64
		#define SEC_SCALAR		1000000000i64
		#define MIN_SCALAR		60000000000i64
		#define HR_SCALAR		3600000000000i64
		#define	DAY_SCALAR		86400000000000i64
		#pragma warning ( disable : 4522 )
		#pragma warning ( disable : 4996 )		// sprintf warning
	#endif
	#ifdef __linux__
		#define MSEC_SCALAR		1000000LL
		#define SEC_SCALAR		1000000000LL
		#define MIN_SCALAR		60000000000LL
		#define HR_SCALAR		3600000000000LL
		#define	DAY_SCALAR		86400000000000LL
	#endif
	#ifdef __CYGWIN__	
		#define MSEC_SCALAR		1000000LL
		#define SEC_SCALAR		1000000000LL
		#define MIN_SCALAR		60000000000LL
		#define HR_SCALAR		3600000000000LL
		#define	DAY_SCALAR		86400000000000LL
	#endif	

	#define ACC_SEC			0
	#define ACC_MSEC		1
	#define ACC_NSEC		2

	#define NSEC_SCALAR		1

	class TimeX {
	public:
		TimeX ();
		TimeX ( sjtime t )			{ m_CurrTime = t; }
		TimeX ( int sec, int msec )	{ m_CurrTime = 0; SetTime ( sec, msec ); }
		
		// Set time
		bool SetTime ( int sec );									// Set seconds
		bool SetTime ( int sec, int msec );							// Set seconds, msecs	
		bool SetTime ( int hr, int min, int m, int d, int y);		// Set hr/min, month, day, year	
		bool SetTime ( int hr, int min, int m, int d, int y, int s, int ms, int ns);	// Set hr/min, month, day, year, sec, ms, ns
		bool SetTime ( TimeX& t )	{ m_CurrTime = t.GetSJT(); return true;} // Set to another Time object				
		bool SetTime ( std::string line );							// Set time from string (hr,min,sec)
		bool SetDate ( std::string line );							// Set date from string (mo,day,yr)		
		void SetSystemTime ();										// Set date/time to system clock		
		void SetTimeNSec ();			

		static sjtime GetSystemMSec ();
		static sjtime GetSystemNSec ();

		void SetSJT ( sjtime t )	{ m_CurrTime = t ;}		// Set Scaled Julian Time directly
		
		// Get time		
		void GetTime (int& sec, int& msec, int& nsec );
		void GetTime (int& hr, int& min, int& m, int& d, int& y);				
		void GetTime (int& hr, int& min, int& m, int& d, int& y, int& s, int& ms, int& ns);
		double GetSec ();
		double GetMSec ();
		std::string GetReadableDate ();
		std::string GetReadableTime ();		
		std::string GetReadableTime ( int fmt );
		std::string GetReadableSJT ();
		std::string GetDayOfWeekName ();
		sjtime GetSJT ()			{ return m_CurrTime; } 			

		// Advance Time
		void Advance ( TimeX& t );
		void AdvanceMinutes ( int n);
		void AdvanceHours ( int n );
		void AdvanceDays ( int n );
		void AdvanceSec ( int n );
		void AdvanceMins ( int n);
		void AdvanceMSec ( int n );
		
		// Utility functions 
		// (these do the actual work, but should not be private as they may be useful to user)
		sjtime GetScaledJulianTime ( int hr, int min, int m, int d, int y );
		sjtime GetScaledJulianTime ( int hr, int min, int m, int d, int y, int s, int ms, int ns );
		void GetTime ( sjtime t, int& hr, int& min, int& m, int& d, int& y);
		void GetTime ( sjtime t, int& hr, int& min, int& m, int& d, int& y, int& s, int& ms, int& ns);
		
		// Get/Set Julian Date and Modified Julain Date
		void SetJD ( double jd );
		void SetMJD ( int jd );
		double GetJD ();
		int GetMJD ();

		// Time operators
		TimeX& operator= ( const TimeX& op );
		TimeX& operator= ( TimeX& op );
		bool operator< ( const TimeX& op );
		bool operator< ( TimeX& op );
		bool operator> ( const TimeX& op );
		bool operator> ( TimeX& op );		
		bool operator<= ( const TimeX& op );
		bool operator<= ( TimeX& op );
		bool operator>= ( const TimeX& op );
		bool operator>= ( TimeX& op );	
		bool operator== ( const TimeX& op );
		bool operator!= ( TimeX& op );
		TimeX operator- ( TimeX& op );
		TimeX operator+ ( TimeX& op );

		// Elapsed Times
		int GetElapsedDays ( TimeX& base );		
		int GetElapsedWeeks ( TimeX& base );
		int GetElapsedMonths ( TimeX& base );
		int GetElapsedYears ( TimeX& base );
		long GetFracDay ( TimeX& base );			// Return Unit = 5 mins
		long GetFracWeek ( TimeX& base );			// Return Unit = 1 hr
		long GetFracMonth ( TimeX& base );		// Return Unit = 4 hrs
		long GetFracYear ( TimeX& base );			// Return Unit = 1 day
		int GetDayOfWeek ();
		int GetWeekOfYear ();

		void RegressionTest ();

	private:		
		static const int	m_DaysInMonth[13];
		static bool			m_Started;

		sjtime				m_CurrTime;		
	};

	// Used for precise system time (Win32)
	void start_timing ( sjtime base );

	//---------------- END TIMING CLASS
#endif

