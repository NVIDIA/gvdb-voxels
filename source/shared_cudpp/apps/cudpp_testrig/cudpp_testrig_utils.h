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
 * cudpp_testrig_utils.h
 *
 */

#ifndef __CUDPP_TESTRIG_UTILS_H__
#define __CUDPP_TESTRIG_UTILS_H__

#define NO_MINMAX
#include <math.h>
#include <cstdio>
#include <iostream>
#include <climits>
#include <float.h>
#include <algorithm>

#include "commandline.h"

using namespace std;
using namespace cudpp_app;

/* Make sure this tracks CUDPPDatatype in cudpp.h! */
inline const char * datatypeToString(CUDPPDatatype t)
{
    static const char * d2s[] =
    {
        "char",
        "uchar",
        "short",
        "ushort",
        "int",
        "uint",
        "float",
        "double",
        "longlong",
        "ulonglong",
        "datatype_invalid",
    };
    return d2s[(int)t];
}

// template specializations defined below after class definitions

template <typename T>
class VectorSupport
{
public:
    static void fillVectorKeys(T *a, size_t numElements, unsigned int keybits);
    static void fillVector(T *a, size_t numElements, float range);
    static int verifySort(T *keysSorted, unsigned int *valuesSorted, 
                          T *keysUnsorted, size_t len, bool reverse);
};

template <typename T>
class OperatorAdd
{
public:
    T operator()(const T& a, const T& b) { return a + b; }
    T identity() { return (T)0; }
};

template <typename T>
class OperatorMultiply
{
public:
    T operator()(const T& a, const T& b) { return a * b; }
    T identity() { return (T)1; }
};

template <typename T>
class OperatorMax
{
public:
    T operator() (const T& a, const T& b) const { return std::max<T>(a, b); }
    T identity() const { return (T)0; }
};

template <typename T>
class OperatorMin
{
public:
    T operator() (const T& a, const T& b) const { return std::min<T>(a, b); }
    T identity() const { return (T)0; }
};

// specializations
template <> inline
int OperatorMax<int>::identity() const { return INT_MIN; }
template <> inline
unsigned int OperatorMax<unsigned int>::identity() const { return 0; }
template <> inline
float OperatorMax<float>::identity() const { return -FLT_MAX; }
template <> inline
double OperatorMax<double>::identity() const { return -DBL_MAX; }
template <> inline
long long OperatorMax<long long>::identity() const { return LLONG_MIN; }
template <> inline
unsigned long long OperatorMax<unsigned long long>::identity() const { return 0; }

template <> inline
int OperatorMin<int>::identity() const { return INT_MAX; }
template <> inline
unsigned int OperatorMin<unsigned int>::identity() const { return UINT_MAX; }
template <> inline
float OperatorMin<float>::identity() const { return FLT_MAX; }
template <> inline
double OperatorMin<double>::identity() const { return DBL_MAX; }
template <> inline
long long OperatorMin<long long>::identity() const { return LLONG_MAX; }
template <> inline
unsigned long long OperatorMin<unsigned long long>::identity() const { return ULLONG_MAX; }
    
template<> inline
void VectorSupport<unsigned int>::fillVectorKeys(unsigned int *a, size_t numElements, unsigned int keybits)
{
    // Fill up with some random data
    int keyshiftmask = 0;
    if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
    int keymask = 0xffff;
    if (keybits < 16) keymask = (1 << keybits) - 1;

    srand(95123);
    for(unsigned int i=0; i < numElements; ++i)   
    { 
        a[i] = ((rand() & keyshiftmask)<<16) | (rand() & keymask); 
    }
}

template<> inline
void VectorSupport<unsigned long long>::fillVectorKeys(unsigned long long *a, size_t numElements, unsigned int keybits)
{
    // Fill up with some random data
    unsigned long long keyshiftmask16 = 0;
    if (keybits > 16) keyshiftmask16 = ((1 << (keybits - 16)) - 1) & 0xffff;
    unsigned long long keyshiftmask32 = 0;
    if (keybits > 32) keyshiftmask32 = ((1 << (keybits - 32)) - 1) & 0xffff;
    unsigned long long keyshiftmask48 = 0;
    if (keybits > 48) keyshiftmask48 = ((1 << (keybits - 48)) - 1) & 0xffff;
    unsigned long long keymask = 0xffff;
    if (keybits < 16) keymask = (1 << keybits) - 1;

    srand(95123);
    for(unsigned int i=0; i < numElements; ++i)   
    { 
        a[i] =
          ((rand() & keyshiftmask48)<<48) |
          ((rand() & keyshiftmask32)<<32) |
          ((rand() & keyshiftmask16)<<16) |
          (rand() & keymask); 
    }
}

template<> inline
void VectorSupport<char>::fillVectorKeys(char *a, size_t numElements, unsigned int  keybits)
{
    VectorSupport<unsigned int>::fillVectorKeys((unsigned int *)a, 
                                                numElements, 
                                                keybits);
}

template<> inline
void VectorSupport<unsigned char>::fillVectorKeys(unsigned char *a, size_t numElements, unsigned int keybits)
{
    VectorSupport<unsigned int>::fillVectorKeys((unsigned int *)a, 
                                                numElements, 
                                                keybits);
}

template<> inline
void VectorSupport<int>::fillVectorKeys(int *a, size_t numElements, unsigned int keybits)
{
    VectorSupport<unsigned int>::fillVectorKeys((unsigned int *)a, 
                                                numElements, 
                                                keybits);
}

template<> inline
void VectorSupport<long long>::fillVectorKeys(long long *a, size_t numElements, unsigned int keybits)
{
    VectorSupport<unsigned long long>::fillVectorKeys((unsigned long long *)a, 
                                                      numElements, 
                                                      keybits);
}

template<> inline
void VectorSupport<float>::fillVectorKeys(float *a, size_t numElements, unsigned int keybits)
{
    VectorSupport<unsigned int>::fillVectorKeys((unsigned int *)a, 
                                                numElements, 
                                                keybits);
}

template<> inline
void VectorSupport<double>::fillVectorKeys(double *a, size_t numElements, unsigned int keybits)
{
    VectorSupport<unsigned long long>::fillVectorKeys((unsigned long long *)a, 
                                                      numElements, 
                                                      keybits);
}



template<> inline
void VectorSupport<float>::fillVector(float *a, size_t numElements, float range)
{
    srand(95123);
    for(size_t j = 0; j < numElements; j++)
    {
        a[j] = pow(-1, (float)j) * (range * (rand() / (float)RAND_MAX));
    }
}

// "info" is the range
template<> inline
void VectorSupport<double>::fillVector(double *a, size_t numElements, float range)
{
    srand(95123);
    for(size_t j = 0; j < numElements; j++)
    {
        a[j] = pow(-1, (double)j) * (range * (rand() / (double)RAND_MAX));
    }
}

template<> inline
void VectorSupport<unsigned char>::fillVector(unsigned char *a, size_t numElements, float range)
{
    srand(95123);
    for(size_t j = 0; j < numElements; j++)
    {
        a[j] = (unsigned char)(range * (rand() / (double)RAND_MAX));
    }
}

template<> inline
void VectorSupport<char>::fillVector(char *a, size_t numElements, float range)
{
    VectorSupport<unsigned char>::fillVector((unsigned char*)a, numElements, range);
}

template<> inline
void VectorSupport<unsigned short>::fillVector(unsigned short *a, size_t numElements, float range)
{
    srand(95123);
    for(size_t j = 0; j < numElements; j++)
    {
        a[j] = (unsigned short)(range * (rand() / (double)RAND_MAX));
    }
}

template<> inline
void VectorSupport<short>::fillVector(short *a, size_t numElements, float range)
{
    VectorSupport<unsigned short>::fillVector((unsigned short*)a, numElements, range);
}

template<> inline
void VectorSupport<unsigned int>::fillVector(unsigned int *a, size_t numElements, float range)
{
    srand(95123);
    for(size_t j = 0; j < numElements; j++)
    {
        a[j] = (unsigned int)(range * (rand() / (double)RAND_MAX));
    }
}

template<> inline
void VectorSupport<int>::fillVector(int *a, size_t numElements, float range)
{
    VectorSupport<unsigned int>::fillVector((unsigned int*)a, numElements, range);
}

template<> inline
void VectorSupport<unsigned long long>::fillVector(unsigned long long *a, size_t numElements, float range)
{
    srand(95123);
    for(size_t j = 0; j < numElements; j++)
    {
        a[j] = (unsigned long long)(range * (rand() / (double)RAND_MAX));
    }
}

template<> inline
void VectorSupport<long long>::fillVector(long long *a, size_t numElements, float range)
{
    VectorSupport<unsigned long long>::fillVector((unsigned long long*)a, numElements, range);
}


// assumes the values were initially indices into the array, for simplicity of 
// checking correct order of values
template<typename T> inline
int VectorSupport<T>::verifySort(T *keysSorted, unsigned int *valuesSorted, 
                                 T *keysUnsorted, size_t len, bool reverse)
{
    int retval = 0;

    for(unsigned int i=0; i<len-1; ++i)
    {   
        bool unordered = reverse ? (keysSorted[i])<(keysSorted[i+1]) 
                                 : (keysSorted[i])>(keysSorted[i+1]);
        if (unordered)
        {
            cout << "Unordered key[" << i << "]:" << keysSorted[i] 
                 << (reverse ? " < " : " > ") << "key["
                 << i+1 << "]:" << keysSorted[i+1] << endl;
            retval = 1;
            break;
        }               
    }

    if (valuesSorted)
    {
        for(unsigned int i=0; i<len; ++i)
        {
            if( keysUnsorted[valuesSorted[i]] != keysSorted[i] )
            {
                cout << "Incorrectly sorted value[" << i << "] ("
                     << valuesSorted[i] << ") " << keysUnsorted[valuesSorted[i]] 
                     << " != " << keysSorted[i] << endl;
                retval = 1;
                break;
            }
        }
    }

    return retval;
}

template<class T> inline
void printItem(T item, const char * sep)
{
    item = item;                // avoid compiler error
    printf("?%s", sep);
}

inline void printItem(float item, const char * sep)
{
    printf("%f%s", item, sep);
}

inline void printItem(int item, const char * sep)
{
    printf("%d%s", item, sep);
}

inline void printItem(double item, const char * sep)
{
    printf("%g%s", item, sep);
}

// http://stackoverflow.com/questions/2844/how-do-you-printf-an-unsigned-long-long-int
inline void printItem(long long item, const char * sep)
{
    printf("%lld%s", item, sep);
}

inline void printItem(unsigned long long item, const char * sep)
{
    printf("%llu%s", item, sep);
}


template<class T>
void printArray(const T * vector, unsigned int len)
{
    for (unsigned int i = 0; i < len; i++)
    { 
        printItem(vector[i], " ");
    }
    printf("\n");
}

inline CUDPPDatatype getDatatypeFromArgv(int argc, const char ** argv)
{
    // check if the command line argument exists
    if( checkCommandLineFlag(argc, argv, "float")) 
    {
        return CUDPP_FLOAT;
    } 
    else if( checkCommandLineFlag(argc, argv, "double")) 
    {
        return CUDPP_DOUBLE;
    }
    if( checkCommandLineFlag(argc, argv, "uint")) 
    {
        return CUDPP_UINT;
    }
    if( checkCommandLineFlag(argc, argv, "int")) 
    {
        return CUDPP_INT;
    }
    if( checkCommandLineFlag(argc, argv, "longlong")) 
    {
        return CUDPP_LONGLONG;
    }
    if( checkCommandLineFlag(argc, argv, "ulonglong")) 
    {
        return CUDPP_ULONGLONG;
    }

    return CUDPP_FLOAT;
}


#endif // __CUDPP_TESTRIG_UTILS_H__


// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
