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
 * tools.cpp
 *
 * @brief Various utilities that are used in cudpp_testrig.
 */

#ifdef CUDPP_APP_COMMON_IMPL

namespace cudpp_app {

    //Dir/File searching routines
    //Since file searching is heavily OS dependent, 
    //Our includes will be different based on which OS we choose
    #if defined (__linux__) || defined (__APPLE__) || defined (MACOSX)
    //used dirent.h to read directories, POSIX standard
    #include <dirent.h>
    #include<unistd.h>
    #elif defined (WIN32) || defined (__WIN32) || defined (WIN64) || defined (__WIN64)
    #pragma warning (disable: 4996) // disable strtok safety warning
    //use io.h and direct.h to read files, this is the windows-version
    #include <io.h>
    #include <direct.h>
    #else
    // error unimplemented OS functionality
    #error "No implementation for this OS in tools.cpp"
    #endif  //#if defined (__linux__) || defined (__APPLE__) || defined (MACOSX)

    #include <stdio.h>
    #include <string.h>

    //function bodies for dir/file searching
    //functions shared by all operating systems

    #if defined (__linux__) || defined (__APPLE__) || defined (MACOSX)
    //linux functions here

    //checks to see if we are currently in the directory to search for
    int checkWorkingDirName(const char * path, const char * target)
    {
        //find the start of the target in path
        char * start;
        char * lastStart = NULL;
        char newPath[500];

        strcpy(newPath, path);
    
        strtok(newPath, "/");
        while((start = strtok(NULL, "/")) != NULL) lastStart = start;
        //effect: find the last directory (i.e. current directory name)

        if(lastStart == NULL) return 0;
        return (strstr(lastStart, target) !=NULL);
    }//end checkWorkingDirName

    int goUpDir(const char * startDir, char * parentPath)
    {
        int val=1;
        char oldDir[500];
        getcwd(oldDir, 500);

        chdir(startDir);
        chdir("..");

        getcwd(parentPath, 500);

        if(strcmp(startDir, parentPath) == 0) val = 0;

        chdir(oldDir);
        return val;
    }

    int gotoParent(const char * parentDirName, char * parentPath)
    {
        int atRoot = 0;
        char cwd[500];
        char parent[500];
        getcwd(cwd,500);
        
        //check the working dir path first
        if(checkWorkingDirName(cwd, parentDirName))
        {
            strcpy(parentPath, cwd);
            return 1;
        }
        while(!atRoot)
        {
            int hitRoot = goUpDir(cwd,parent);
            if(checkWorkingDirName(parent, parentDirName))
            {
                strcpy(parentPath, parent);
                return 1;
            }
            strcpy(cwd, parent);
            atRoot = !hitRoot;
        }
        return 0;
    }
    void getRootPath(char * path, int size)
    {
        getcwd(path, size);
    }
    //finds out how high up our target directory is
    int cutupPath(const char * target)
    {
        char path[500];
        getcwd(path, 500);  //linux / apple version is getcwd, while windows is _getcwd...

        if(checkWorkingDirName(path, target))
            return 0;

        char * curDir = strtok(path, "/");
        char * token;
    
        int counter = 0;
        bool haveSeen = false;
    
        if(strcmp(curDir, target) == 0)
        {
            haveSeen = true;
        } //end if
    
        while((token = strtok(NULL, "/")) != NULL)
        {           
            if(haveSeen)
                counter++;
            
            if(strcmp(token, target) == 0)
            {
                haveSeen = true;
            }
        }  //end while
    
        if(haveSeen)
            return counter;
        else
            return -1;
    }  //end cutupPath

    void constructHomeDirBasePath(char * pathName, int numUp)
    {
        pathName[0] = '\0';

        //check a base case: if numUp = 0, then path is .    
        if(numUp == 0)
        {
            strcat(pathName, ".");
            return;
        }//end if

        for(int i=0; i<numUp-1; i++)
            strcat(pathName, "../");
        
        strcat(pathName, "..");     
    }//end constructHomeDirBasePath

    int findDirWithBase(const char * base, const char * dirName, char * outputPath)
    {
        DIR *dir;
        char newPath[500];

        if((dir = opendir (base)) == NULL)
        {
            printf( "Unable to open %s\n", base );
            return 0;
        }//end if
        else
        {
            dirent * dp;
            while ((dp = readdir(dir)) != NULL) 
            {
                //first check, make sure it is not any of the directories we don't 
                //want to search
                if(strcmp(dp->d_name, "..") == 0 || 
                   strcmp(dp->d_name, ".") == 0  || 
                   strcmp(dp->d_name, ".svn") == 0)
                    continue;

                DIR * d;
                //need to concatenate
                sprintf(newPath, "%s/%s", base, dp->d_name);

                if((d = opendir(newPath))==NULL)
                {
                    //not a directory, do nothing for the directory search case
                }//end if
                else
                {
                    //we found another directory!
                    //check first to see if this dir matches the one we are looking for
                    if(strcmp(dp->d_name, dirName) == 0)
                    {
                        //this is a match!
                        strcpy(outputPath, newPath);
                        strcat(outputPath, "/");
                        closedir(d);
                        return 1;
                    }//end if
                    //recurse
                    int result = findDirWithBase(newPath, dirName, outputPath);

                    closedir(d);
                    if(result)
                        return 1;
                }//end else
            }//end while ((dp = readdir(dir)) != NULL) 
        }  //end else
        closedir(dir);
        //didn't find it
        return 0;
    }//void findDirWithBase

    int findPaths(const char * parentDirName, char paths[10][500])
    {
        int numPathsFound=0;
        int atRoot = 0;
        char cwd[500];
        char parent[500];
        getcwd(cwd,500);

        //check the working dir path first
        if(checkWorkingDirName(cwd, parentDirName))
        {
            strcpy(paths[numPathsFound], cwd);
            numPathsFound++;
        }

        while(!atRoot)
        {
            int hitRoot = goUpDir(cwd,parent);
            if(checkWorkingDirName(parent, parentDirName))
            {
                strcpy(paths[numPathsFound], parent);
                numPathsFound++;
            }
            strcpy(cwd, parent);
            atRoot = !hitRoot;
        }
        return numPathsFound;
    }

    int findFileWithBase(const char * base, const char * fileName, char * outputPath)
    {
        DIR *dir;
        char newPath[500];
    
        if((dir = opendir (base)) == NULL)
        {
            printf( "Unable to open %s\n", base );
            return 0;
        }//end if
        else
        {
            dirent * dp;
            while ((dp = readdir(dir)) != NULL) 
            {
                //first check, make sure it is not any of the directories we don't 
                //want to search
                if(strcmp(dp->d_name, "..") == 0 || 
                   strcmp(dp->d_name, ".") == 0  || 
                   strcmp(dp->d_name, ".svn") == 0)
                    continue;
            
                DIR * d;
                //need to concatenate
                sprintf(newPath, "%s/%s", base, dp->d_name);
                if((d = opendir(newPath))==NULL)
                {
                    //this is a file, check with the fileName
                    //check first to see if this dir matches the one we are looking for
                    if(strcmp(dp->d_name, fileName) == 0)
                    {
                        //this is a match!
                        strcpy(outputPath, newPath);
                        return 1;
                    }//end if
                
                }//end if((d = opendir(newPath))==NULL)
                else
                {
                    //recurse
                    int result = findFileWithBase(newPath, fileName, outputPath);

                    closedir(d);
                    if(result)
                        return 1;
                }//end else
            }//end while ((dp = readdir(dir)) != NULL)
        }//end else  

        closedir(dir);
        //didn't find it
        return 0;
    }//void findFileWithBase

    #elif defined (WIN32) || defined (__WIN32) || defined (WIN64) || defined (__WIN64)
    //windows functions here
    //checks to see if we are currently in the directory to search for
    int checkWorkingDirName(const char * path, const char * target)
    {
        //find the start of the target in path
        char * start;
        char * lastStart = NULL;
        char newPath[500];

        strcpy(newPath, path);
    
        strtok(newPath, "\\");
        while((start = strtok(NULL, "\\")) != NULL) lastStart = start;
        //effect: find the last directory (i.e. current directory name)

        if(lastStart == NULL) return 0;
        return (lastStart != 0 && strstr(lastStart, target) !=NULL);
    }//end checkWorkingDirName

    int goUpDir(const char * startDir, char * parentPath)
    {
        int val=1;
        char oldDir[500];
        _getcwd(oldDir, 500);

        _chdir(startDir);
        _chdir("..");

        _getcwd(parentPath, 500);

        if(strcmp(startDir, parentPath) == 0) val = 0;

        _chdir(oldDir);
        return val;
    }
    int gotoParent(const char * parentDirName, char * parentPath)
    {
        int atRoot = 0;
        char cwd[500];
        char parent[500];
        _getcwd(cwd,500);

        //check the working dir path first
        if(checkWorkingDirName(cwd, parentDirName))
        {
            strcpy(parentPath, cwd);
            return 1;
        }
        while(!atRoot)
        {
            int hitRoot = goUpDir(cwd,parent);
            if(checkWorkingDirName(parent, parentDirName))
            {
                strcpy(parentPath, parent);
                return 1;
            }
            strcpy(cwd, parent);
            atRoot = !hitRoot;
        }
        return 0;
    }

    int findPaths(const char * parentDirName, char paths[10][500])
    {
        int numPathsFound=0;
        int atRoot = 0;
        char cwd[500];
        char parent[500];
        _getcwd(cwd,500);

        //check the working dir path first
        if(checkWorkingDirName(cwd, parentDirName))
        {
            strcpy(paths[numPathsFound], cwd);
            numPathsFound++;
        }
        while(!atRoot)
        {
            int hitRoot = goUpDir(cwd,parent);
            if(checkWorkingDirName(parent, parentDirName))
            {
                strcpy(paths[numPathsFound], parent);
                numPathsFound++;
            }
            strcpy(cwd, parent);
            atRoot = !hitRoot;
        }
        return numPathsFound;
    }

    void getRootPath(char * rootPath, int size)
    {
        _getcwd(rootPath, size);
    }
    int cutupPath(const char * target)
    {
        char path[500];
        _getcwd(path, 500);

        if(checkWorkingDirName(path, target))
            return 0;

        char * curDir = strtok(path, "\\");
        char * token;
    
        int counter = 0;
        bool haveSeen = false;
    
        if(curDir != 0 && strcmp(curDir, target) == 0)
        {
            haveSeen = true;
        }
    
        while((token = strtok(NULL, "\\")) != NULL)
        {           
            if(haveSeen)
                counter++;
            
            if(token != 0 && strcmp(token, target) == 0)
                haveSeen = true;

        }//end while
    
        if(haveSeen)
            return counter;
        else
            return -1;
    }  //end cutupPath


    void constructHomeDirBasePath(char * pathName, int numUp)
    {
        pathName[0] = '\0';

        //check a base case: if numUp = 0, then path is .    
        if(numUp == 0)
        {
            strcat(pathName, ".");
            return;
        }//end if

        if(numUp < 0)
            return;

        for(int i=0; i<numUp-1; i++)
            strcat(pathName, "..\\");
        
        strcat(pathName, "..");     
    }  //end constructHomeDirBasePath

    int findDirWithBase(const char * base, const char * dirName, char * outputPath)
    {
        _finddata_t c_file;
        intptr_t hFile;

        char newPath[500];
        char baseCpy[500];
        strcpy(baseCpy, base);
        strcat(baseCpy, "\\*");
    
        if( (hFile = _findfirst( baseCpy, &c_file )) == -1L )
        {
            printf( "Unable to open %s\n", base );
            return 0;
        }//end if 
        else
        {
            do {
                //first check, make sure it is not any of the directories we don't 
                //want to search
                if(strcmp(c_file.name, "..") == 0 ||
                   strcmp(c_file.name, ".") == 0 ||
                   strcmp(c_file.name, ".svn") == 0)
                    continue;

                if(c_file.attrib & _A_SUBDIR)
                {
                    //construct the new path
                    sprintf(newPath, "%s\\%s", base, c_file.name);

                    //this is a directory, see if it matches the name we are looking for
                    if(strcmp(dirName, c_file.name) == 0)
                    {
                        //this is the dir we are looking for
                        strcpy(outputPath, newPath);
                        strcat(outputPath, "\\");
                        _findclose( hFile );
                        return 1;
                    }//end if
                    //this is not the directory we are looking for (jedi powers...)
                    //recurse
                    int result = findDirWithBase(newPath, dirName, outputPath);
                    if(result)
                    {
                        _findclose(hFile);
                        return 1;
                    }  //end if(result)

                }//end if(c_file.attrib & _A_SUBDIR)
            } while( _findnext( hFile, &c_file ) == 0 );
            _findclose( hFile );
        }//end else

        return 0;
    }  //end findDirWithBase

    int findFileWithBase(const char * base, const char * fileName, char * outputFile)
    {
        _finddata_t c_file;
        intptr_t hFile;

        char newPath[500];
        char baseCpy[500];
        strcpy(baseCpy, base);
        strcat(baseCpy, "\\*");
        // Find first .c file in current directory 
        if( (hFile = _findfirst( baseCpy, &c_file )) == -1L )
        {
            printf( "Unable to open %s\n", base );
            return 0;
        }//end if
        else
        {
            do {
                //first check, make sure it is not any of the directories we don't 
                //want to search
                if(strcmp(c_file.name, "..") == 0 ||
                   strcmp(c_file.name, ".") == 0 ||
                   strcmp(c_file.name, ".svn") == 0)
                    continue;

                if(c_file.attrib & _A_SUBDIR)
                {
                    //construct the new path
                    sprintf(newPath, "%s\\%s", base, c_file.name);
                    //recurse
                    int result = findFileWithBase(newPath, fileName, outputFile);
                    if(result)
                    {
                        _findclose(hFile);
                        return 1;
                    }//end if(result)

                }//end if
                else
                {
                    //check the file to see if it matches the name
                    if(strcmp(fileName, c_file.name) == 0)
                    {
                        //this is the dir we are looking for
                        sprintf(outputFile, "%s\\%s", base, c_file.name);
                        _findclose( hFile );
                        return 1;
                    }  //end if
                }  //end else
            } while( _findnext( hFile, &c_file ) == 0 );
            _findclose( hFile );
        }//end else

        return 0;
    }//end findFileWithBase
    #else
    // error unimplemented OS functionality
    #error "No implementation for this OS in tools.cpp"

    #endif //#if defined (__linux__) || defined (__APPLE__) || defined (MACOSX)


    //dir/file functions generic to both OSes (i.e. the wrapper functions)

    /**@brief Attempts to find a directory starting the the root directory \a startDir which is named
     *   \a dirName, final relative path (if found) is stored in \a outputPath
     *
     * This function tries to find a directory called \a dirName starting at a parent directory called \a startDir
     * It returns 1 if successful and when successful \a outputPath contains the relative path from the current working
     * directory to \a dirName.  If no such directory is found, 0 is returned and nothing is written to \a outputPath
     *
     * @param[out] outputPath the relative path from the current working directory to the directory in question.  
     * @param[in] startDir the starting directory to begin searching for \a dirName
     * @param[in] dirName the directory in question
     */
    bool findDir(const char * startDir, const char * dirName, char * outputPath)
    {
        char rootPath[500];
    //      char parentPath[500];
        char pathsFound[10][500];
        getRootPath(rootPath, 500);
        // fprintf(stderr, "finding number of possible dirs...\n");
        int numPossibleDirs = findPaths(startDir, pathsFound);
        // fprintf(stderr, "numPossibleDirs: %d\n", numPossibleDirs);

        if (numPossibleDirs ==0) return false;

        int val;
        for(int i=numPossibleDirs-1; i >=0; i--)
            // for(int i=0; i<numPossibleDirs; i++)
        {
            val =findDirWithBase(pathsFound[i], dirName,outputPath);
            if(val == 1)
                return true;
        }

        return false;
    //      printf("starting at %s and looking for %s\n", rootPath, startDir);
    //      int val = gotoParent(startDir, parentPath);


    //      printf("found base: %s\n", parentPath);
        //   return findDirWithBase(parentPath, dirName,outputPath);
    }//end int findDir(char * startDir, char * dirName, char * outputPath)

    /**@brief Attempts to find a file starting in the root directory \a startDir which is named
     *   \a fileName, final relative path of the file (if found) is stored in \a outputPath
     *
     * This function tries to find a file called \a fileName starting at a parent directory called \a startDir
     * It returns 1 if successful and when successful \a outputPath contains the relative path from the current working
     * directory to \a fileName.  If no such file is found, 0 is returned and nothing is written to \a outputPath
     *
     * @param[out] outputPath the relative path from the current working directory to the directory in question.  
     * @param[in] startDir the starting directory to begin searching for \a dirName
     * @param[in] fileName the file in question
     */
    bool findFile(const char * startDir, const char * fileName, char * outputPath)
    {
        char rootPath[500];
    //      char parentPath[500];
        char pathsFound[10][500];
        getRootPath(rootPath, 500);
        int numPossibleDirs = findPaths(startDir, pathsFound);

        if (numPossibleDirs ==0) return false;

        int val;
        for(int i=0; i<numPossibleDirs; i++)
        {
            val =findFileWithBase(pathsFound[i], fileName,outputPath);
            if(val == 1)
                return true;
        }

        return false;
    }//end int findFile(char * startDir, char * fileName, char * outputPath)

}

#endif // CUDPP_APP_COMMON_IMPL

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
