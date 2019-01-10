// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// -------------------------------------------------------------

#include <cudpp.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include "cudpp_testrig_utils.h"
#include "sparse.h"
#define MAX_ALPHA 256

typedef unsigned int uint;

bool leq(int a1, int a2, int b1, int b2) // lexicographic order
{ return(a1 < b1 || a1 == b1 && a2 <= b2); } // for pairs

bool leq(int a1, int a2, int a3, int b1, int b2, int b3)
{ return(a1 < b1 || a1 == b1 && leq(a2,a3, b2,b3)); } // and triples

// stably sort a[0..n-1] to b[0..n-1] with keys in 0..K from r
void radixPass(uint* a, uint* b, uint* r, int n, int K)
{ // count occurrences
    uint* c = new uint[K + 1]; // counter array
    for (int i = 0; i <= K; i++) c[i] = 0; // reset counters
    for (int i = 0; i < n; i++) c[r[a[i]]]++; // count occurrences
    for (int i = 0, sum = 0; i <= K; i++) // exclusive prefix sums
    { int t = c[i]; c[i] = sum; sum += t; }
    for (int i = 0; i < n; i++) b[c[r[a[i]]]++] = a[i]; // sort
    delete [] c;
}

// find the suffix array SA of T[0..n-1] in {1..K}^n
// require T[n]=T[n+1]=T[n+2]=0, n>=2
void suffixArray(uint* T, uint* SA, int n, int K) {
    int n0=(n+2)/3, n1=(n+1)/3, n2=n/3, n02=n0+n2;
    uint* R = new uint[n02 + 3]; R[n02]= R[n02+1]= R[n02+2]=0;
    uint* SA12 = new uint[n02 + 3]; SA12[n02]=SA12[n02+1]=SA12[n02+2]=0;
    uint* R0 = new uint[n0];
    uint* SA0 = new uint[n0];

    //******* Step 0: Construct sample ********
    // generate positions of mod 1 and mod 2 suffixes
    // the "+(n0-n1)" adds a dummy mod 1 suffix if n%3 == 1
    for (int i=0, j=0; i < n+(n0-n1); i++) if (i%3 != 0) R[j++] = i;
    //******* Step 1: Sort sample suffixes ********
    // lsb radix sort the mod 1 and mod 2 triples
    radixPass(R , SA12, T+2, n02, K);
    radixPass(SA12, R , T+1, n02, K);
    radixPass(R , SA12, T , n02, K);
    // find lexicographic names of triples and
    // write them to correct places in R
    int name = 0, c0 = -1, c1 = -1, c2 = -1;
    for (int i = 0; i < n02; i++) {
        if (T[SA12[i]] != c0 || T[SA12[i]+1] != c1 || T[SA12[i]+2] != c2)
        { name++; c0 = T[SA12[i]]; c1 = T[SA12[i]+1]; c2 = T[SA12[i]+2]; }
        if (SA12[i] % 3 == 1) { R[SA12[i]/3] = name; } // write to R1
        else { R[SA12[i]/3 + n0] = name; } // write to R2
    }
    // recurse if names are not yet unique
    if (name < n02) {
        suffixArray(R, SA12, n02, name);
        // store unique names in R using the suffix array
        for (int i = 0; i < n02; i++) R[SA12[i]] = i + 1;
    } else // generate the suffix array of R directly
        for (int i = 0; i < n02; i++) SA12[R[i] - 1] = i;

    //******* Step 2: Sort nonsample suffixes ********
    // stably sort the mod 0 suffixes from SA12 by their first character
    for (int i=0, j=0; i < n02; i++) if (SA12[i] < n0) R0[j++] = 3*SA12[i];
    radixPass(R0, SA0, T, n0, K);

    //******* Step 3: Merge ********
    // merge sorted SA0 suffixes and sorted SA12 suffixes
    for (int p=0, t=n0-n1, k=0; k < n; k++) {
#define GetI() (SA12[t] < n0 ? SA12[t] * 3 + 1 : (SA12[t] - n0) * 3 + 2)
        int i = GetI(); // pos of current offset 12 suffix
        int j = SA0[p]; // pos of current offset 0 suffix
        if (SA12[t] < n0 ? // different compares for mod 1 and mod 2 suffixes
                leq(T[i], R[SA12[t] + n0], T[j], R[j/3]) :
                leq(T[i],T[i+1],R[SA12[t]-n0+1], T[j],T[j+1],R[j/3+n0]))
        { // suffix from SA12 is smaller
            SA[k] = i; t++;
            if (t == n02) // done --- only SA0 suffixes left
                for (k++; p < n0; p++, k++) SA[k] = SA0[p];
        } else { // suffix from SA0 is smaller
            SA[k] = j; p++;
            if (p == n0) // done --- only SA12 suffixes left
                for (k++; t < n02; t++, k++) SA[k] = GetI();
        }
    }
    delete [] R; delete [] SA12; delete [] SA0; delete [] R0;
}


////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set for suffix-array
//! Each element is a position of a suffix.
//! @param idata      const input data as provided to device
//! @param reference  reference data, computed but preallocated
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void
computeSaGold(unsigned char* idata, unsigned int* reference, size_t len)
{
    unsigned int *inp = new unsigned int[len+3];
    for(int i=0; i<len; ++i) inp[i] = (unsigned int) idata[i] + 1;
    inp[len]=0; inp[len+1]=0; inp[len+2]=0;
    for(int i=0; i<len+3; ++i) reference[i]=0;
    suffixArray(inp, reference, len, MAX_ALPHA);
    delete [] inp;
}


// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
