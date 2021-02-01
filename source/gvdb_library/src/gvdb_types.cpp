//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
// Version 1.1: Rama Hoetzlein, 3/25/2018
//-----------------------------------------------------------------------------

#include "gvdb_types.h"

#include <cstdlib>
#include <sstream>
#include <stdio.h>
#include <stdarg.h>

//------------------------------------------------------- gprintf
static size_t fmt2_sz    = 0;
static char *fmt2 = NULL;
static FILE *fd = NULL;
static bool bLogReady = false;
static bool bPrintLogging = true;
static int  printLevel = -1; // <0 mean no level prefix

void gprintSetLevel(int l)
{
    printLevel = l;
}
int gprintGetLevel()
{
    return printLevel;
}
void gprintSetLogging(bool b)
{
    bPrintLogging = b;
}
void gprintf2(va_list &vlist, const char * fmt, int level)
{
	if(bPrintLogging == false)
        return;
    if(fmt2_sz == 0) {
        fmt2_sz = 1024;
        fmt2 = (char*)malloc(fmt2_sz);
    }
    while((vsnprintf(fmt2, fmt2_sz, fmt, vlist)) < 0) // means there wasn't anough room
    {
        fmt2_sz *= 2;
        if(fmt2) free(fmt2);
        fmt2 = (char*)malloc(fmt2_sz);
    }
    char *prefix = "";
    switch(level)
    {
    case LOGLEVEL_WARNING:
        prefix = "LOG *WARNING* >> ";
		return;
        break;
    case LOGLEVEL_ERROR:
        prefix = "LOG **ERROR** >> ";
        break;
    case LOGLEVEL_OK:
        prefix = "LOG !OK! >> ";
        break;
    case LOGLEVEL_INFO:
    default:
        break;
    }
    ::printf(prefix);
    ::printf(fmt2);
}
void gprintf(const char * fmt, ...)
{
//    int r = 0;
    va_list  vlist;
    va_start(vlist, fmt);
    gprintf2(vlist, fmt, printLevel);
	va_end(vlist);
}
void gprintfLevel(int level, const char * fmt, ...)
{
    va_list  vlist;
    va_start(vlist, fmt);
    gprintf2(vlist, fmt, level);
	va_end(vlist);
}

void gerror ()
{
	gprintf ( "Error. Application will exit.\n" );	
	exit(-1);
}
