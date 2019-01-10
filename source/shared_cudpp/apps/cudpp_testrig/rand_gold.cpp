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
 * rand_gold.cu
 *
 * @brief Host testrig routines to execute MD5
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

namespace testrig {

    struct uint4
    {
        unsigned int x, y, z, w;
    };

    struct dim3
    {
        unsigned int x, y, z;
    };

    void swizzleShift(uint4 *f)
    {
        unsigned int temp;
        temp = f->x;
        f->x = f->y;
        f->y = f->z;
        f->z = f->w;
        f->w = temp;
    }

    unsigned int leftRotate(unsigned int x, unsigned int n)
    {
        unsigned int t = ( ((x) << (n)) | ((x) >> (32-n)) ) ;
        return t;
    }

    unsigned int F(unsigned int x, unsigned int y, unsigned int z)
    {
        unsigned int t;
        t = ( (x&y) | ((~x) & z) );
        return t;
    }

    unsigned int G(unsigned int x, unsigned int y, unsigned int z)
    {
        unsigned int t;
        t = ( (x&z) | ((~z) & y) );
        return t;
    }

    unsigned int H(unsigned int x, unsigned int y, unsigned int z)
    {
        unsigned int t;
        t = (x ^ y ^ z );
        return t;
    }
    unsigned int I(unsigned int x, unsigned int y, unsigned int z)
    {
        unsigned int t;
        t = ( y ^ (x | ~z) );
        return t;
    }

    void FF(uint4 * td, int i, uint4 * Fr, float p, unsigned int * data)
    {
        unsigned int Ft = F(td->y, td->z, td->w);
        unsigned int r = Fr->x;
        swizzleShift(Fr);
    
        float t = sin((float)(i)) * p;
        unsigned int trigFunc = (unsigned int)(t);
        td->x = td->y + leftRotate(td->x + Ft + trigFunc + data[i], r);
        swizzleShift(td);
    }

    void GG(uint4 * td, int i, uint4 * Gr, float p, unsigned int * data)
    {
        unsigned int Ft = G(td->y, td->z, td->w);
        i = (5*i+1) %16;
        unsigned int r = Gr->x;
        swizzleShift(Gr);
    
        float t = sin(float(i)) * p;
        unsigned int trigFunc = (unsigned int)(t);
        td->x = td->y + leftRotate(td->x + Ft + trigFunc + data[i], r);
        swizzleShift(td);
    }

    void HH(uint4 * td, int i, uint4 * Hr, float p, unsigned int * data)
    {
        unsigned int Ft = H(td->y, td->z, td->w);
        i = (3*i+5) %16;
        unsigned int r = Hr->x;
        swizzleShift(Hr);
    
        float t = sin(float(i)) * p;
        unsigned int trigFunc = (unsigned int)(t);
        td->x = td->y + leftRotate(td->x + Ft + trigFunc + data[i], r);
        swizzleShift(td);
    }

    void II(uint4 * td, int i, uint4 * Ir, float p, unsigned int * data)
    {
        unsigned int Ft = G(td->y, td->z, td->w);
        i = (7*i) %16;
        unsigned int r = Ir->x;
        swizzleShift(Ir);
    
        float t = sin(float(i)) * p;
        unsigned int trigFunc = (unsigned int)(t);
        td->x = td->y + leftRotate(td->x + Ft + trigFunc + data[i], r);
        swizzleShift(td);
    }


    void setupInput(unsigned int * input, unsigned int seed, dim3 threadIdx, dim3 blockIdx, dim3 blockDim)
    {	
        //loop unroll, also do this more intelligently
        input[0] = threadIdx.x ^ seed;
        input[1] = threadIdx.y ^ seed;
        input[2] = threadIdx.z ^ seed;
        input[3] = 0x80000000 ^ seed;
        input[4] = blockIdx.x ^ seed;
        input[5] = blockIdx.y ^ seed;
        input[6] = blockIdx.z ^ seed;
        input[7] = blockDim.x ^ seed;
        input[8] = blockDim.y ^ seed;
        input[9] = blockDim.z ^ seed;
        input[10] = seed;
        input[11] = seed;
        input[12] = seed;
        input[13] = seed;
        input[14] = seed;
        input[15] = 128 ^ seed;
    }

    void gen_randMD5CPU(uint4 *d_out, size_t numElements, unsigned int seed,dim3 threadIdx, dim3 blockIdx, dim3 blockDim)
    {
        unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int data[16];
        setupInput(data, seed, threadIdx, blockIdx, blockDim);	//meaning we fix this at the same time
    
        unsigned int h0 = 0x67452301;
        unsigned int h1 = 0xEFCDAB89;
        unsigned int h2 = 0x98BADCFE;
        unsigned int h3 = 0x10325476;

        uint4 result = { h0,h1,h2,h3 };
        uint4 td = result;

        float p = powf(2.0f,32.0f);
    
        uint4 Fr = { 7,12,17,22 };
        uint4 Gr = { 5,9,14,20 };
        uint4 Hr = { 4,11,16,23 };
        uint4 Ir = { 6,10,15,21 };	
    
        //for optimization, this is loop unrolled
        FF(&td, 0, &Fr,p,data);
        FF(&td, 1, &Fr,p,data);
        FF(&td, 2, &Fr,p,data);
        FF(&td, 3, &Fr,p,data);
        FF(&td, 4, &Fr,p,data);
        FF(&td, 5, &Fr,p,data);
        FF(&td, 6, &Fr,p,data);
        FF(&td, 7, &Fr,p,data);
        FF(&td, 8, &Fr,p,data);
        FF(&td, 9, &Fr,p,data);
        FF(&td,10, &Fr,p,data);
        FF(&td,11, &Fr,p,data);
        FF(&td,12, &Fr,p,data);
        FF(&td,13, &Fr,p,data);
        FF(&td,14, &Fr,p,data);
        FF(&td,15, &Fr,p,data);
    
        GG(&td,16, &Gr,p,data);
        GG(&td,17, &Gr,p,data);
        GG(&td,18, &Gr,p,data);
        GG(&td,19, &Gr,p,data);
        GG(&td,20, &Gr,p,data);
        GG(&td,21, &Gr,p,data);
        GG(&td,22, &Gr,p,data);
        GG(&td,23, &Gr,p,data);
        GG(&td,24, &Gr,p,data);
        GG(&td,25, &Gr,p,data);
        GG(&td,26, &Gr,p,data);
        GG(&td,27, &Gr,p,data);
        GG(&td,28, &Gr,p,data);
        GG(&td,29, &Gr,p,data);
        GG(&td,30, &Gr,p,data);
        GG(&td,31, &Gr,p,data);
    
        HH(&td,32, &Hr,p,data);
        HH(&td,33, &Hr,p,data);
        HH(&td,34, &Hr,p,data);
        HH(&td,35, &Hr,p,data);
        HH(&td,36, &Hr,p,data);
        HH(&td,37, &Hr,p,data);
        HH(&td,38, &Hr,p,data);
        HH(&td,39, &Hr,p,data);
        HH(&td,40, &Hr,p,data);
        HH(&td,41, &Hr,p,data);
        HH(&td,42, &Hr,p,data);
        HH(&td,43, &Hr,p,data);
        HH(&td,44, &Hr,p,data);
        HH(&td,45, &Hr,p,data);
        HH(&td,46, &Hr,p,data);
        HH(&td,47, &Hr,p,data);

        II(&td,48, &Ir,p,data);
        II(&td,49, &Ir,p,data);
        II(&td,50, &Ir,p,data);
        II(&td,51, &Ir,p,data);
        II(&td,52, &Ir,p,data);
        II(&td,53, &Ir,p,data);
        II(&td,54, &Ir,p,data);
        II(&td,55, &Ir,p,data);
        II(&td,56, &Ir,p,data);
        II(&td,57, &Ir,p,data);
        II(&td,58, &Ir,p,data);
        II(&td,59, &Ir,p,data);
        II(&td,60, &Ir,p,data);
        II(&td,61, &Ir,p,data);
        II(&td,62, &Ir,p,data);
        II(&td,63, &Ir,p,data);
        
        result.x = result.x + td.x;
        result.y = result.y + td.y;
        result.z = result.z + td.z;
        result.w = result.w + td.w;
    
        //printf("computed result: %u %u %u %u\n", result.x, result.y, result.z, result.w);
        //printf("idx is: %d and numElements: %d\n", idx, numElements);
        if (idx < numElements)
            d_out[idx] = result;
    }

    void randMD5CPUDispatch(unsigned int * data, size_t numElements, unsigned int seed)
    {
        //compute the new element size for the uint4 array

        unsigned int newSize = (unsigned int)numElements / 4;

        newSize += (numElements %4 == 0) ? 0:1;

        uint4 * h_out = (uint4 *) malloc(sizeof(uint4) * newSize);

        //now we need to manually calculate the block size and such
        dim3 blockSize, blockID, threadID;
    
        //default blockSize is 256, but if there aren't enough elements, for this, then blockSize is the elements
        blockSize.x = (newSize > 256) ? 256 : newSize;
        blockSize.y = blockSize.z = 1;
        blockID.x = blockID.y = blockID.z = 1;
        blockID.x = 0;
        threadID.x = threadID.y = threadID.z = 1;

        for(unsigned int i = 0; i < newSize; i += blockSize.x)
        {
            //go through each block and give it a threadID
            for(unsigned int j = 0; j < blockSize.x; j++)
            {
                threadID.x = i + j;
                gen_randMD5CPU(h_out, newSize, seed, threadID, blockID, blockSize);
            }//end for j

            blockID.x++;
        }//end for i

        //now copy all the generated elements back into data
        unsigned int j = 0;
        for(unsigned int i = 0; i < newSize; i++)
        {
            //this is bad code, but I will fix later!
            if (j >=numElements) break;
            data[j++] = h_out[i].x;
            if (j >=numElements) break;
            data[j++] = h_out[i].y;
            if (j >=numElements) break;
            data[j++] = h_out[i].z;
            if (j >=numElements) break;
            data[j++] = h_out[i].w;
        }
        //debug use only: print out the rands:

        for(unsigned int i = 0; i < newSize; i++)
            printf("%u %u %u %u\n", h_out[i].x,h_out[i].y,h_out[i].z,h_out[i].w);

        printf("\n\n");
    }//end randMD5CPUDispatch
}

