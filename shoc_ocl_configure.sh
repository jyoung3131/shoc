#/bin/bash
#Configure SHOC using CPU or GPU OpenCL includes and libraries
. setenv_ocl.sh $1
./configure --with-cuda --with-opencl CPPFLAGS="-I$OCL_INCL" LDFLAGS="-L$OCL_LIB" LIBS="-lpthread $LIBS"
