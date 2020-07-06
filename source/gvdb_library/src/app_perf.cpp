

/* APP_PERF

Copyright 2009-2012  NVIDIA Corporation.  All rights reserved.

R. Hoetzlein
This lightweight performance class provides additional
features for CPU and GPU profiling:

1. By default, markers are disabled if nvToolsExt32_1.dll 
   is not found in the working path. Useful when shipping a product.
2. Providing nvToolsExt32_1.dll automatically enables CPU and GPU markers.
3. If nvToolsExt_32.dll is not present, you can still 
   enable CPU only markers by calling PERF_INIT(false);  // false = don't require dll
4. Instrument code with single PERF_PUSH, PERF_POP markers 
   for both CPU and GPU output.
5. Perform additional printing along with markers using PERF_PRINTF
6. Output CPU markers to log file by specifing filename to PERF_SET
7. Markers can be nested, with range output for both CPU and GPU
8. Only app_perf.h and app_perf.cpp are needed. No other dependencies. 
   No need to link with nvToolsExt.h.
9. CPU and GPU can be enabled selectively in different parts of the app.
   Call PERF_SET( CPUon?, CPUlevel, GPUon?, LogFilename ) at any time.
10. CPU Level specifies maximum printf level for markers. Useful when
    your markers are inside an inner loop. You can keep in code, but hide their output.
11. GPU markers use NVIDIA's Perfmarkers for viewing in NVIDIA NSIGHT

*/

/*
* Copyright 2009-2012  NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* This software and the information contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and conditions
* of a form of NVIDIA software license agreement.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.   This source code is a "commercial item" as
* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer  software"  and "commercial computer software
* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*
* Any use of this source code in individual and commercial software must
* include, in the user documentation and internal comments to the code,
* the above Disclaimer and U.S. Government End Users Notice.
*/

#include "app_perf.h"

#ifdef _WIN32
#	include <conio.h>
#	include <io.h>
#else
	#ifdef USE_NVTX
		#include <nvToolsExt.h>
    #endif 
#endif
#include <cstring>
#include <fcntl.h>	
#include <cstdlib>

extern void				gprintf(const char * fmt, ...);
#define PERF_PRINTF		gprintf

bool				g_perfInit = false;			// Is perf started? Checks for DLL

#ifdef USE_NVTX
	nvtxRangePushFunc	g_nvtxPush = 0x0;			// Pointer to nv-perfmarker func
	nvtxRangePopFunc	g_nvtxPop = 0x0;			// Pointer to nv-perfmarker func
#else
	char*				g_nvtxPush = 0x0;
	char*				g_nvtxPop = 0x0;
#endif

int					g_perfLevel = 0;			// Current level of push/pop
sjtime				g_perfStack[1024];			// Stack of recorded CPU timings
char				g_perfMsg[1024][256];			// Stack of recorded messages
FILE*				g_perfCons = 0x0;			// On-screen console to output CPU timing
int					g_perfPrintLev = 2;			// Maximum level to print. Set with PERF_SET
bool				g_perfCPU = false;			// Do CPU timing? Set with PERF_SET
bool				g_perfGPU = false;			// Do GPU timing? Set with PERF_SET
bool				g_perfConsOut = true;
std::string			g_perfFName = "";			// File name for CPU output. Set with PERF_SET
FILE*				g_perfFile = 0x0;			// File handle for output

void PERF_START ()
{
	g_perfStack [ g_perfLevel ] = TimeX::GetSystemNSec ();
	g_perfLevel++;
}

float PERF_STOP ()
{
	g_perfLevel--;
	sjtime curr = TimeX::GetSystemNSec ();
	curr -= g_perfStack [ g_perfLevel ];
	return ((float) curr) / MSEC_SCALAR;
}


void PERF_PUSH ( const char* msg )
{
	#ifdef USE_NVTX
		if ( g_perfGPU ) (*g_nvtxPush) (msg);	
	#endif
	if ( g_perfCPU ) {
		if ( ++g_perfLevel < g_perfPrintLev ) {
			strncpy ( (char*) g_perfMsg[ g_perfLevel ], msg, 256 );
			g_perfStack [ g_perfLevel ] = TimeX::GetSystemNSec ();				
			if ( g_perfConsOut ) PERF_PRINTF ( "%*s%s\n", g_perfLevel <<1, "", msg );
			if ( g_perfFile != 0x0 ) fprintf ( g_perfFile, "%*s%s\n", g_perfLevel <<1, "", msg );
		}		
	}
}
float PERF_POP ()
{
	#ifdef USE_NVTX
		if ( g_perfGPU ) (*g_nvtxPop) ();
	#endif
	if ( g_perfCPU ) {
		if ( g_perfLevel < g_perfPrintLev) {
			sjtime curr = TimeX::GetSystemNSec ();
			curr -= g_perfStack [ g_perfLevel ];		
			if ( g_perfConsOut ) PERF_PRINTF ( "%*s%s: %f ms\n", g_perfLevel <<1, "", g_perfMsg[g_perfLevel], ((float) curr)/MSEC_SCALAR );		
			if ( g_perfFile != 0x0 ) fprintf ( g_perfFile, "%*s%s: %f ms\n", g_perfLevel <<1, "", g_perfMsg[g_perfLevel], ((float) curr)/MSEC_SCALAR );
			g_perfLevel--;
			return ((float) curr) / MSEC_SCALAR;
		}	
		g_perfLevel--;
	}
	return 0;
}


void PERF_SET ( bool cons, int lev )
{
	g_perfConsOut = cons;
	if ( lev == 0 ) lev = 32767;
	g_perfPrintLev = lev;
}

void PERF_INIT ( int buildbits, bool cpu, bool gpu, bool cons, int lev, const char* fname )
{
	g_perfCPU = cpu;
	g_perfGPU = gpu;
	g_perfConsOut = cons;
	if ( lev == 0 ) lev = 32767;
	g_perfPrintLev = lev;
	g_perfInit = true;
	g_perfLevel = 0;	
	g_perfFile = 0x0;
	g_perfFName = fname;	
	if ( g_perfFName.length() > 0 ) {
		if ( g_perfFile == 0x0 ) g_perfFile = fopen ( g_perfFName.c_str(), "wt" );
	}
	
	#if defined(WIN32)
		// Address of NV Perfmarker functions	
		char libname[128];	
		if ( buildbits == 64 ) {
			strcpy ( libname, "nvToolsExt64_1.dll" );		
		} else {
			strcpy ( libname, "nvToolsExt32_1.dll" );
		}

		#ifdef USE_NVTX

			#ifdef UNICODE
				wchar_t libwc[128];
				MultiByteToWideChar(CP_ACP, 0, libname, -1, libwc, 8192);   		
				LoadLibrary ( libwc );  		
				HMODULE mod = GetModuleHandle( libwc );
			#else
				LoadLibrary ( libname );  		
				HMODULE mod = GetModuleHandle( libname );
			#endif	
			g_nvtxPush = (nvtxRangePushFunc) GetProcAddress( mod, "nvtxRangePushA");
			g_nvtxPop  = (nvtxRangePopFunc)  GetProcAddress( mod, "nvtxRangePop");
		#else
			if (g_perfConsOut) PERF_PRINTF("WARNING: GPU markers not enabled for GVDB Library. Set cmake flag USE_NVTX.\n");
			g_perfGPU = false;
		#endif

		// Console window for CPU timings
		if ( g_perfCPU ) {
			AllocConsole ();
			HANDLE lStdHandle = GetStdHandle( STD_OUTPUT_HANDLE );
			int hConHandle = _open_osfhandle((intptr_t)lStdHandle, _O_TEXT);
			g_perfCons = _fdopen( hConHandle, "w" );
			setvbuf(g_perfCons, NULL, _IONBF, 1);
			*stdout = *g_perfCons;
			PERF_PRINTF ( "PERF_INIT: Enabling CPU markers.\n" );
		} else {
			PERF_PRINTF ( "PERF_INIT: No CPU markers.\n" );
		}
		if ( g_perfGPU ) {
			if ( g_nvtxPush != 0x0 && g_nvtxPop != 0x0 ) {
				PERF_PRINTF ( "PERF_INIT: Enabling GPU markers. Found %s.\n", libname );			
			} else {			
				PERF_PRINTF ( "PERF_INIT: Disabling GPU markers. Did not find %s.\n", libname );			
				g_perfGPU = false;
			}		
		} else {
			PERF_PRINTF ( "PERF_INIT: No GPU markers.\n" );
		}
		if ( !g_perfCPU && !g_perfGPU ) {
			PERF_PRINTF ( "PERF_INIT: Disabling perf. No CPU or GPU markers.\n" );			
		}
	#else
		#ifdef USE_NVTX
			g_nvtxPush = nvtxRangePushA;
			g_nvtxPop = nvtxRangePop;
		#endif
		g_perfCons = 0x0;		
	#endif

	TimeX start;		// create Time obj to initialize system timer
}


//---------------- TIMING CLASS
// R.Hoetzlein

#ifdef _MSC_VER
	#include <windows.h>
	#include <mmsystem.h>
#else
	#include <sys/time.h>
#endif 

#include <stdio.h>
#include <time.h>
#include <math.h>

#ifdef _MSC_VER
	#define VS2005
	#pragma comment ( lib, "winmm.lib" )
	LARGE_INTEGER	m_BaseCount;
	LARGE_INTEGER	m_BaseFreq;
#endif

const int TimeX::m_DaysInMonth[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
bool TimeX::m_Started = false;
sjtime			m_BaseTime;
sjtime			m_BaseTicks;

void start_timing ( sjtime base )
{	
	m_BaseTime = base;

	#ifdef _MSC_VER
		m_BaseTicks = timeGetTime();
		QueryPerformanceCounter ( &m_BaseCount );
		QueryPerformanceFrequency ( &m_BaseFreq );
	#else
		struct timeval tv;
		gettimeofday(&tv, NULL);
		m_BaseTicks = ((sjtime) tv.tv_sec * 1000000LL) + (sjtime) tv.tv_usec;		
	#endif
}

sjtime TimeX::GetSystemMSec ()
{
	#ifdef _MSC_VER
		return m_BaseTime + sjtime(timeGetTime() - m_BaseTicks)*MSEC_SCALAR;
	#else
		struct timeval tv;
		gettimeofday(&tv, NULL);			
		sjtime t = ((sjtime) tv.tv_sec * 1000000LL) + (sjtime) tv.tv_usec;	
		return m_BaseTime + ( t - m_BaseTicks) * 1000LL;			// 1000LL - converts microseconds to milliseconds
	#endif
}

sjtime TimeX::GetSystemNSec ()
{
	#ifdef _MSC_VER
		LARGE_INTEGER currCount;
		QueryPerformanceCounter ( &currCount );
		return m_BaseTime + sjtime( (double(currCount.QuadPart-m_BaseCount.QuadPart) / m_BaseFreq.QuadPart) * SEC_SCALAR);
	#else
		printf ( "ERROR: GetSystemNSec not implemented. QueryPerformanceCounter not available.\n" );
	#endif	
}

void TimeX::SetTimeNSec ()
{
	m_CurrTime = GetSystemNSec ();
}


TimeX::TimeX ()
{	
	if ( !m_Started ) {
		m_Started = true;			
		SetSystemTime ();				// Get base time from wall clock
		start_timing ( m_CurrTime );	// Start timing from base time		
	}
	m_CurrTime = 0;
}

// Note regarding hours:
//  0 <= hr <= 23
//  hr = 0 is midnight (12 am)
//  hr = 1 is 1 am
//  hr = 12 is noon 
//  hr = 13 is 1 pm (subtract 12)
//  hr = 23 is 11 pm (subtact 12)

// GetScaledJulianTime
// Returns -1.0 if the time specified is invalid.
sjtime TimeX::GetScaledJulianTime ( int hr, int min, int m, int d, int y, int s, int ms, int ns )
{
	double MJD;				// Modified Julian Date (JD - 2400000.5)
	sjtime SJT;				// Scaled Julian Time SJT = MJD * 86400000 + UT

	// Check if date/time is valid
	if (m <=0 || m > 12) return (sjtime) -1;	
	if ( y % 4 == 0 && m == 2) {	// leap year in february
		if (d <=0 || d > m_DaysInMonth[m]+1) return (sjtime) -1;
	} else {
		if (d <=0 || d > m_DaysInMonth[m]) return (sjtime) -1;		
	}
	if (hr < 0 || hr > 23) return (sjtime) -1;
	if (min < 0 || min > 59) return  (sjtime) -1;

	// Compute Modified Julian Date
	MJD = 367 * y - int ( 7 * (y + int (( m + 9)/12)) / 4 );
	MJD -= int ( 3 * (int((y + (m - 9)/7)/100) + 1) / 4);
	MJD += int ( 275 * m / 9 ) + d + 1721028.5 - 1.0;
	MJD -= 2400000.5;
	// Compute Scaled Julian Time
	SJT = sjtime(MJD) * sjtime( DAY_SCALAR );	
	SJT += hr * HR_SCALAR + min * MIN_SCALAR + s * SEC_SCALAR + ms * MSEC_SCALAR + ns * NSEC_SCALAR;
	return SJT;
}

sjtime TimeX::GetScaledJulianTime ( int hr, int min, int m, int d, int y )
{
	return GetScaledJulianTime ( hr, min, m, d, y, 0, 0, 0 );
}

void TimeX::GetTime ( sjtime SJT, int& hr, int& min, int& m, int& d, int& y)
{
	int s = 0, ms = 0, ns = 0;
	GetTime ( SJT, hr, min, m, d, y, s, ms, ns );
}

void TimeX::GetTime ( sjtime SJT, int& hr, int& min, int& m, int& d, int& y, int& s, int &ms, int& ns)
{	
	// Compute Universal Time from SJT
	sjtime UT = sjtime( SJT % sjtime( DAY_SCALAR ) );

	// Compute Modified Julian Date from SJT
	double MJD = double(SJT / DAY_SCALAR);

	// Use MJD to get Month, Day, Year
	double z = floor ( MJD + 1 + 2400000.5 - 1721118.5);
	double g = z - 0.25;
	double a = floor ( g / 36524.25 );
	double b = a - floor  ( a / 4.0 );
	y = int( floor (( b + g ) / 365.25 ) );
	double c = b + z - floor  ( 365.25 * y );
	m = int (( 5 * c + 456) / 153 );
	d = int( c - int (( 153 * m - 457) / 5) );
	if (m > 12) {
		y++;
		m -= 12;
	}
	// Use UT to get Hrs, Mins, Secs, Msecs
	hr = int( UT / HR_SCALAR );
	UT -= hr * HR_SCALAR;
	min = int( UT / MIN_SCALAR );
	UT -= min * MIN_SCALAR;
	s = int ( UT / SEC_SCALAR );
	UT -= s * SEC_SCALAR;
	ms = int ( UT / MSEC_SCALAR );	
	UT -= ms * MSEC_SCALAR;
	ns = int ( UT / NSEC_SCALAR );

	// UT Example:
	//      MSEC_SCALAR =         1
	//      SEC_SCALAR =      1,000
	//      MIN_SCALAR =     60,000
	//		HR_SCALAR =   3,600,000
	//      DAY_SCALAR = 86,400,000
	//
	//   7:14:03, 32 msec 	
	//   UT = 7*3,600,000 + 14*60,000 + 3*1,000 + 32 = 26,043,032
	//
	//   26,043,032 / 3,600,000 = 7			26,043,032 - (7 * 3,600,000) = 843,032
	//      843,032 /    60,000 = 14		   843,032 - (14 * 60,000) = 3,032
	//        3,032 /     1,000 = 3              3,032 - (3 * 1,000) = 32
	//           32 /         1 = 32	
}

void TimeX::GetTime (int& s, int& ms, int& ns )
{
	int hr, min, m, d, y;
	GetTime ( m_CurrTime, hr, min, m, d, y, s, ms, ns );
}


void TimeX::GetTime (int& hr, int& min, int& m, int& d, int& y)
{
	GetTime ( m_CurrTime, hr, min, m, d, y);
}

void TimeX::GetTime (int& hr, int& min, int& m, int& d, int& y, int& s, int& ms, int& ns)
{
	GetTime ( m_CurrTime, hr, min, m, d, y, s, ms, ns);
}

bool TimeX::SetTime ( int sec )
{
	int hr, min, m, d, y;
	GetTime ( m_CurrTime, hr, min, m, d, y );
	m_CurrTime = GetScaledJulianTime ( hr, min, m, d, y, sec, 0, 0 );
	return true;
}

bool TimeX::SetTime ( int sec, int msec )
{
	int hr, min, m, d, y;
	GetTime ( m_CurrTime, hr, min, m, d, y );
	m_CurrTime = GetScaledJulianTime ( hr, min, m, d, y, sec, msec, 0 );
	return true;
}

bool TimeX::SetTime (int hr, int min, int m, int d, int y)
{
	int s, ms, ns;
	GetTime ( s, ms, ns );
	m_CurrTime = GetScaledJulianTime ( hr, min, m, d, y, s, ms, ns );
	if (m_CurrTime == -1.0) return false;
	return true;
}

bool TimeX::SetTime (int hr, int min, int m, int d, int y, int s, int ms, int ns)
{
	m_CurrTime = GetScaledJulianTime ( hr, min, m, d, y, s, ms, ns );
	if (m_CurrTime == -1.0) return false;
	return true;
}

bool TimeX::SetTime ( std::string line )
{
	int hr, min, m, d, y;
	std::string dat;
	if ( line.substr ( 0, 1 ) == " " ) 
		dat = line.substr ( 1, line.length()-1 ).c_str();
	else 
		dat = line;

	hr = atoi ( dat.substr ( 0, 2).c_str() );
	min = atoi ( dat.substr ( 3, 2).c_str() );
	m = atoi ( dat.substr ( 6, 2).c_str () );
	d = atoi ( dat.substr ( 9, 2).c_str () );
	y = atoi ( dat.substr ( 12, 4).c_str () );
	return SetTime ( hr, min, m, d, y);
}

bool TimeX::SetDate ( std::string line )
{
	int hr, min, m, d, y;
	std::string dat;
	if ( line.substr ( 0, 1 ) == " " ) 
		dat = line.substr ( 1, line.length()-1 ).c_str();
	else 
		dat = line;

	hr = 0;
	min = 0;
	m = atoi ( dat.substr ( 0, 2).c_str () );
	d = atoi ( dat.substr ( 3, 2).c_str () );
	y = atoi ( dat.substr ( 6, 4).c_str () );
	return SetTime ( hr, min, m, d, y);
}

std::string TimeX::GetDayOfWeekName ()
{
	switch (GetDayOfWeek()) {
	case 1:		return "Sunday";	break;
	case 2:		return "Monday";	break;
	case 3:		return "Tuesday";	break;
	case 4:		return "Wednesday";	break;
	case 5:		return "Thursday";	break;
	case 6:		return "Friday";	break;
	case 7:		return "Saturday";	break;
	}
	return "day error";
}

int TimeX::GetDayOfWeek ()
{
	// Compute Modified Julian Date
	double MJD = (double) m_CurrTime / sjtime( DAY_SCALAR );

	// Compute Julian Date
	double JD = floor ( MJD + 1 + 2400000.5 );
	int dow = (int(JD - 0.5) % 7) + 4;
	if (dow > 7) dow -= 7;

	// day of week (1 = sunday, 7 = saturday)
	return dow ;
}

int TimeX::GetWeekOfYear ()
{
	int hr, min, m, d, y;
	GetTime ( hr, min, m, d, y );
	double mjd_start = (double) GetScaledJulianTime ( 0, 0, 1, 1, y ) / DAY_SCALAR; // mjt for jan 1st of year
	double mjd_curr = (double) GetScaledJulianTime ( 0, 0, m, d, y ) / DAY_SCALAR; // mjt for specified day in year
	double JD = floor ( mjd_start + 1 + 2400000.5 );
	int dow = (int ( JD - 0.5 ) % 7) + 4;  // day of week for jan 1st of year.
	if (dow > 7) dow -= 7;
	
	// week of year (first week in january = week 0)
	return int((mjd_curr - mjd_start + dow -1 ) / 7 );
}

int TimeX::GetElapsedDays ( TimeX& base )
{
	return int( sjtime(m_CurrTime - base.GetSJT() ) / sjtime( DAY_SCALAR ) );
}

int TimeX::GetElapsedWeeks ( TimeX& base )
{
	return GetElapsedDays(base) / 7;
}

int TimeX::GetElapsedMonths ( TimeX& base)
{
	return int ( double(GetElapsedDays(base)) / 30.416 );
}

int TimeX::GetElapsedYears ( TimeX& base )
{
	// It is much easier to compute this in m/d/y format rather
	// than using julian dates.
	int bhr, bmin, bm, bd, by;
	int ehr, emin, em, ed, ey;
	GetTime ( base.GetSJT(), bhr, bmin, bm, bd, by );
	GetTime ( m_CurrTime, ehr, emin, em, ed, ey );
	if ( em < bm) {
		// earlier month
		return ey - by - 1;
	} else if ( em > bm) {
		// later month
		return ey - by;
	} else {
		// same month
		if ( ed < bd ) {
			// earlier day
			return ey - by - 1;
		} else if ( ed >= bd ) {
			// later or same day
			return ey - by;
		}
	}
	return -1;
}

#pragma warning(disable:4244)

long TimeX::GetFracDay ( TimeX& base )
{
	// Resolution = 5-mins
	return long( sjtime(m_CurrTime - base.GetSJT() ) % sjtime(DAY_SCALAR) ) / (MIN_SCALAR*5);
}

long TimeX::GetFracWeek ( TimeX& base )
{
	// Resolution = 1 hr
	long day = GetElapsedDays(base) % 7;		// day in week
	long hrs = long( sjtime(m_CurrTime - base.GetSJT() ) % sjtime(DAY_SCALAR) ) / (HR_SCALAR);
	return day * 24 + hrs;
}

long TimeX::GetFracMonth ( TimeX& base )
{
	// Resolution = 4 hrs
	long day = (long) fmod ( double(GetElapsedDays(base)), 30.416 );	// day in month
	long hrs = long( sjtime(m_CurrTime - base.GetSJT() ) % sjtime(DAY_SCALAR) ) / (HR_SCALAR*4);
	return day * (24 / 4) + hrs;
}

long TimeX::GetFracYear ( TimeX& base )
{
	// It is much easier to compute this in m/d/y format rather
	// than using julian dates.
	int bhr, bmin, bm, bd, by;
	int ehr, emin, em, ed, ey;
	sjtime LastFullYear;
	GetTime ( base.GetSJT() , bhr, bmin, bm, bd, by );
	GetTime ( m_CurrTime, ehr, emin, em, ed, ey );
	if ( em < bm) {
		// earlier month
		LastFullYear = GetScaledJulianTime ( ehr, emin, bm, bd, ey - 1);		
		return long( sjtime(m_CurrTime - LastFullYear) / sjtime(DAY_SCALAR) );		
	} else if ( em > bm) {
		// later month
		LastFullYear = GetScaledJulianTime ( ehr, emin, bm, bd, ey);
		return long( sjtime(m_CurrTime - LastFullYear) / sjtime(DAY_SCALAR) );				
	} else {
		// same month
		if ( ed < bd ) {
			// earlier day
			LastFullYear = GetScaledJulianTime ( ehr, emin, bm, bd, ey - 1);		
			return long( sjtime(m_CurrTime - LastFullYear) / sjtime(DAY_SCALAR) );		
		} else if ( ed > bd ) {
			// later day
			LastFullYear = GetScaledJulianTime ( ehr, emin, bm, bd, ey);
			return long( sjtime(m_CurrTime - LastFullYear) / sjtime(DAY_SCALAR) );
		} else {
			return 0;	// same day
		}
	}	
}

std::string TimeX::GetReadableDate ()
{
	char buf[200];
	std::string line;
	int hr, min, m, d, y;

	GetTime ( hr, min, m, d, y );
	sprintf ( buf, "%02d:%02d %02d-%02d-%04d", hr, min, m, d, y);	
	return std::string ( buf );
}

std::string TimeX::GetReadableTime ()
{
	char buf[200];
	std::string line;
	int hr, min, m, d, y, s, ms, ns;

	GetTime ( hr, min, m, d, y, s, ms, ns );	
	sprintf ( buf, "%02d:%02d,%03d.%06d", min, s, ms, ns);
	//sprintf ( buf, "%02d:%02d:%02d %03d.%06d %02d-%02d-%04d", hr, min, s, ms, ns, m, d, y);
	return std::string ( buf );
}

std::string TimeX::GetReadableSJT ()
{
	char buf[200];	
	sprintf ( buf, "%I64d", m_CurrTime );
	return std::string ( buf );
}

std::string TimeX::GetReadableTime ( int fmt )
{
	char buf[200];	
	int hr, min, m, d, y, s, ms, ns;
	GetTime ( hr, min, m, d, y, s, ms, ns );	

	switch (fmt) {
	case 0: sprintf ( buf, "%02d %03d.%06d", s, ms, ns);
	}
	return std::string ( buf );
}

void TimeX::SetSystemTime ()
{
	int hr, mn, sec, m, d, y;
	char timebuf[100];
	char datebuf[100];
	std::string line;

	#ifdef _MSC_VER
		#ifdef VS2005
		_strtime_s ( timebuf, 100 );
		_strdate_s ( datebuf, 100 );
		#else
		_strtime ( timebuf );
		_strdate ( datebuf );
		#endif
	#endif
	#if (defined(__linux__) || defined(__CYGWIN__))
		time_t tt; 
		struct tm tim;
		tt = time(NULL);	
		localtime_r(&tt, &tim);	
		sprintf( timebuf, "%02i:%02i:%02i", tim.tm_hour, tim.tm_min, tim.tm_sec);
		sprintf( datebuf, "%02i:%02i:%02i", tim.tm_mon, tim.tm_mday, tim.tm_year % 100);
	#endif

	line = timebuf;
	hr = atoi ( line.substr ( 0, 2).c_str() );
	mn = atoi ( line.substr ( 3, 2).c_str() );
	sec = atoi ( line.substr ( 6, 2).c_str() );
	line = datebuf;
	m = atoi ( line.substr ( 0, 2).c_str() );
	d = atoi ( line.substr ( 3, 2).c_str() );
	y = atoi ( line.substr ( 6, 2).c_str() );
	
	// NOTE: This only works from 1930 to 2030
	if ( y > 30) y += 1900;
	else y += 2000;
	
	SetTime ( hr, mn, m, d, y, sec, 0, 0);
}
   

double TimeX::GetSec ()
{
	return ((double) m_CurrTime / (double) SEC_SCALAR );
}

double TimeX::GetMSec ()
{
	return ((double) m_CurrTime / (double) MSEC_SCALAR );

	/*int s, ms, ns;
	GetTime ( s, ms, ns );
	return ms;*/
}

void TimeX::Advance ( TimeX& t )
{
	m_CurrTime += t.GetSJT ();
}

void TimeX::AdvanceMinutes ( int n)
{
	m_CurrTime += (sjtime) MIN_SCALAR * n;
}

void TimeX::AdvanceHours ( int n )
{
	m_CurrTime += (sjtime) HR_SCALAR * n;	
}

void TimeX::AdvanceDays ( int n )
{
	m_CurrTime += (sjtime) DAY_SCALAR * n;	
}

void TimeX::AdvanceSec ( int n )
{
	m_CurrTime += (sjtime) SEC_SCALAR * n;	
}

void TimeX::AdvanceMins ( int n)
{
	m_CurrTime += (sjtime) MIN_SCALAR * n;
}	

void TimeX::AdvanceMSec ( int n )
{
	m_CurrTime += (sjtime) MSEC_SCALAR * n;	
}

TimeX& TimeX::operator= ( const TimeX& op )	{ m_CurrTime = op.m_CurrTime; return *this; }
TimeX& TimeX::operator= ( TimeX& op )		{ m_CurrTime = op.m_CurrTime; return *this; }
bool TimeX::operator< ( const TimeX& op )	{ return (m_CurrTime < op.m_CurrTime); }
bool TimeX::operator> ( const TimeX& op )	{ return (m_CurrTime > op.m_CurrTime); }
bool TimeX::operator< ( TimeX& op )			{ return (m_CurrTime < op.m_CurrTime); }
bool TimeX::operator> ( TimeX& op )			{ return (m_CurrTime > op.m_CurrTime); }

bool TimeX::operator<= ( const TimeX& op )		{ return (m_CurrTime <= op.m_CurrTime); }
bool TimeX::operator>= ( const TimeX& op )		{ return (m_CurrTime >= op.m_CurrTime); }
bool TimeX::operator<= ( TimeX& op )			{ return (m_CurrTime <= op.m_CurrTime); }
bool TimeX::operator>= ( TimeX& op )			{ return (m_CurrTime >= op.m_CurrTime); }

TimeX TimeX::operator- ( TimeX& op )
{
	return TimeX( m_CurrTime - op.GetSJT() );
}
TimeX TimeX::operator+ ( TimeX& op )
{
	return TimeX( m_CurrTime + op.GetSJT() );
}

bool TimeX::operator== ( const TimeX& op )
{
	return (m_CurrTime == op.m_CurrTime);
}
bool TimeX::operator!= ( TimeX& op )
{
	return (m_CurrTime != op.m_CurrTime);
}
	
void TimeX::RegressionTest ()
{
	// This code verifies the Julian Date calculations are correct for all
	// minutes over a range of years. Useful to debug type issues when
	// compiling on different platforms.
	//
	int m, d, y, hr, min;
	int cm, cd, cy, chr, cmin;

	for (y=2000; y < 2080; y++) {
		for (m=1; m <= 12; m++) {
			for (d=1; d <= 31; d++) {
				for (hr=0; hr<=23; hr++) {
					for (min=0; min<=59; min++) {
						if ( SetTime ( hr, min, m, d, y, 0, 0, 0 ) ) {
							GetTime ( chr, cmin, cm, cd, cy );
							if ( hr!=chr || min!=cmin || m!=cm || d!=cd || y!=cy) {
//								debug.Printf (" time", "Error: %d, %d, %d, %d, %d = %I64d\n", hr, min, m, d, y, GetSJT());
//								debug.Printf (" time", "-----: %d, %d, %d, %d, %d\n", chr, cmin, cm, cd, cy);
							}
						}
					}
				}				
			}			
		}
//		debug.Printf (" time", "Verified: %d\n", y);
	}
}



