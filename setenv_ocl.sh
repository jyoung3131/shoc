#!/bin/bash
#Allow for switching between Intel, Beignet, or NVIDIA OpenCL libraries
#Last updated: 10/6/2016

USERNAME=`whoami`

#Ubuntu 14.04
#PREVPATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#Ubuntu 16.04
PREVPATH=/home/$USERNAME/bin:/home/$USERNAME/.local/bin:/home/$USERNAME/bin:/home/$USERNAME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:/snap/bin

if [ -z $1 ]
then
        echo "Please enter 1, 2, or 3 (Intel SDK 2016, AMD, CUDA) as an argument"
        exit
fi

if [ $1 == "1" ];
then
        #SDK has include files but /etc/OpenCl/vendors/intel64.icd points to other directory for lib
        export PATH=/opt/intel/intel-opencl-1.2-6.2.0.1760/opencl-1.2-sdk-6.2.0.1760/bin:/opt/intel/intel-opencl-1.2-6.2.0.1760/gt_debugger_2016.0/bin:$PREVPATH
        export LD_LIBRARY_PATH=/opt/intel/intel-opencl-1.2-6.2.0.1760/opencl-1.2-sdk-6.2.0.1760/lib64:/usr/local/lib:/usr/lib64:/usr/lib
        export OCL_INCL=/opt/intel/intel-opencl-1.2-6.2.0.1760/opencl-1.2-sdk-6.2.0.1760/include
elif [ $1 == "2" ];
then
        export PATH=:$PREVPATH
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
	export OCL_INCL=
elif [ $1 == "3" ];
then
        export PATH=/usr/local/cuda/bin:$PREVPATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/lib:/usr/lib64:/usr/lib
        export OCL_LIB=/usr/local/cuda/lib64
        export OCL_INCL=/usr/local/cuda/include
fi

echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "OCL_LIB: $OCL_LIB"
echo "OCL_INCL: $OCL_INCL"
