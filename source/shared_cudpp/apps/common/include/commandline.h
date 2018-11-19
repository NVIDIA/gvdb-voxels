// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * commandline.h
 * 
 * @brief Command line argument parsing
 * 
 */


#ifndef _COMMAND_LINE_H_
#define _COMMAND_LINE_H_

#include <string>
#include <sstream>

namespace cudpp_app {

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line argument arrays
    //! @note This function is used each type for which no template specialization
    //!  exist (which will cause errors if the type does not fulfill the std::vector
    //!  interface).
    ////////////////////////////////////////////////////////////////////////////////
    template<class T>
    inline T convertTo( const std::string& element)
    {
        (void) element;  // suppress compiler warning
        return (T)false;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line arguments of type char
    ////////////////////////////////////////////////////////////////////////////////
    template<>
    inline char convertTo<char>( const std::string& element) 
    {
        std::istringstream ios( element);
        char val;
        ios >> val;
        return val;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line arguments of type unsigned char
    ////////////////////////////////////////////////////////////////////////////////
    template<>
    inline unsigned char convertTo<unsigned char>( const std::string& element) 
    {
        std::istringstream ios( element);
        char val;
        ios >> val;
        return val;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line arguments of type int
    ////////////////////////////////////////////////////////////////////////////////
    template<>
    inline int convertTo<int>( const std::string& element) 
    {
        std::istringstream ios( element);
        int val;
        ios >> val;
        return val;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line arguments of type unsigned int
    ////////////////////////////////////////////////////////////////////////////////
    template<>
    inline unsigned int convertTo<unsigned int>( const std::string& element) 
    {
        std::istringstream ios( element);
        int val;
        ios >> val;
        return val;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line arguments of type long long
    ////////////////////////////////////////////////////////////////////////////////
    template<>
    inline long long convertTo<long long>( const std::string& element) 
    {
        std::istringstream ios( element);
        long long val;
        ios >> val;
        return val;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line arguments of type unsigned long long
    ////////////////////////////////////////////////////////////////////////////////
    template<>
    inline unsigned long long convertTo<unsigned long long>( const std::string& element) 
    {
        std::istringstream ios( element);
        long long val;
        ios >> val;
        return val;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line arguments of type float
    ////////////////////////////////////////////////////////////////////////////////
    template<>
    inline float convertTo<float>( const std::string& element) 
    {
        std::istringstream ios( element);
        float val;
        ios >> val;
        return val;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line arguments of type double
    ////////////////////////////////////////////////////////////////////////////////
    template<>
    inline double convertTo<double>( const std::string& element) 
    {
        std::istringstream ios( element);
        double val;
        ios >> val;
        return val;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line arguments of type string
    ////////////////////////////////////////////////////////////////////////////////
    template<>
    inline std::string convertTo<std::string>( const std::string& element)
    {
        return element;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line arguments of type bool
    ////////////////////////////////////////////////////////////////////////////////
    template<>
    inline bool convertTo<bool>( const std::string& element) 
    {
        // check if value is given as string-type { true | false }
        if ( "true" == element) 
            return true;
        else if ( "false" == element) 
            return false;
        // check if argument is given as integer { 0 | 1 }
        else 
        {
            int tmp = convertTo<int>( element ); 
            return ( 1 == tmp);
        }

        return false;
    }

    template <typename T>
    bool commandLineArg(T& val, int argc, const char**argv, const char* name)
    {
        for( int i=1; i<argc; ++i) 
        {
            std::string arg = argv[i];
            size_t pos = arg.find(name);
            if (pos != std::string::npos && pos == 1 && arg[0] == '-') {
                std::string::size_type pos;
                // check if only flag or if a value is given
                if ( (pos = arg.find( '=')) == std::string::npos) 
                {  
                    val = convertTo<T>("true");                                  
                    return true;
                }
                else 
                {
                    val = convertTo<T>(std::string( arg, pos+1, arg.length()-1));
                    return true;
                }
            }
        }
        return false;
    }

    bool checkCommandLineFlag(int argc, const char** argv, const char* name);
}

#ifdef CUDPP_APP_COMMON_IMPL
#include "commandline.inl"
#endif

#endif // #ifndef _COMMAND_LINE_H_

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
