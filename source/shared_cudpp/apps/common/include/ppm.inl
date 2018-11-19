// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Source: $
// $Revision: $
// $Date: $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * ppm.inl
 *
 * @brief PPM/PGM Image loading
 */

#ifdef CUDPP_APP_COMMON_IMPL

#include <iostream>

namespace cudpp_app {
    
    //! size of PGM file header 
    const unsigned int PGMHeaderSize = 0x40;
        
    //////////////////////////////////////////////////////////////////////////////
    //! Load PGM or PPM file
    //! @note if data == NULL then the necessary memory is allocated in the 
    //!       function and w and h are initialized to the size of the image
    //! @return true if the file loading succeeded, otherwise false
    //! @param file        name of the file to load
    //! @param data        handle to the memory for the image file data
    //! @param w        width of the image
    //! @param h        height of the image
    //! @param channels number of channels in image
    //////////////////////////////////////////////////////////////////////////////
    bool
    loadPPM( const char* file, unsigned char** data, 
             unsigned int *w, unsigned int *h, unsigned int *channels ) 
    {
        FILE *fp = NULL;
        if(NULL == (fp = fopen(file, "rb"))) 
        {
            std::cerr << "loadPPM() : Failed to open file: " << file << std::endl;
            return false;
        }

        // check header
        char header[PGMHeaderSize], *string = NULL;
        string = fgets( header, PGMHeaderSize, fp);
        if (strncmp(header, "P5", 2) == 0)
        {
            *channels = 1;
        }
        else if (strncmp(header, "P6", 2) == 0)
        {
            *channels = 3;
        }
        else {
            std::cerr << "loadPPM() : File is not a PPM or PGM image" << std::endl;
            *channels = 0;
            return false;
        }

        // parse header, read maxval, width and height
        unsigned int width = 0;
        unsigned int height = 0;
        unsigned int maxval = 0;
        unsigned int i = 0;
        while(i < 3) 
        {
            string = fgets(header, PGMHeaderSize, fp);
            if(header[0] == '#') 
                continue;

            if(i == 0) 
            {
                i += sscanf( header, "%u %u %u", &width, &height, &maxval);
            }
            else if (i == 1) 
            {
                i += sscanf( header, "%u %u", &height, &maxval);
            }
            else if (i == 2) 
            {
                i += sscanf(header, "%u", &maxval);
            }
        }

        // check if given handle for the data is initialized
        if( NULL != *data) 
        {
            if (*w != width || *h != height) 
            {
                std::cerr << "loadPPM() : Invalid image dimensions." << std::endl;
                return false;
            }
        } 
        else 
        {
            *data = (unsigned char*) malloc( sizeof( unsigned char) * width * height * *channels);
            *w = width;
            *h = height;
        }

        // read and close file
        size_t fsize = 0;
        fsize = fread( *data, sizeof(unsigned char), width * height * *channels, fp);
        fclose(fp);

        return true;
    }
}

#endif