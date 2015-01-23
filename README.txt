This code is a fork of the SHOC benchmark suite to include 
the new OpenCL TPC-H microbenchmark and primitive-based 
benchmarks. At the moment, this code is *ALPHA* and so 
likely has runtime bugs that are under development.

To build:
./configure (with correct OpenCL flags)
make install

To run:
$cd bin/Serial/OpenCl
$./Tpch --test-name <testNm> --num-elems <N> --join-size <J> -n <numPasses> --platform $plNum --device

Valid arguments are:
test-name: Select, Project, Join, Unique, A, B, C
num-elems: 256 - 8388608 (4K - 128 MB)
join-size: 1-4 where 4 produces the largest join output
n: Number of iterations for a particular benchmark
platform: The OpenCL platform - typically 0 or 1 (e.g., CPU vs. GPU)
device: 0-4 depending on the number of CPU/GPU/MIC in the system

=========================================

The Scalable HeterOgeneous Computing (SHOC) benchmark suite is a
collection of benchmark programs testing the performance and
stability of systems using computing devices with non-traditional architectures
for general purpose computing. Its initial focus is on systems containing
Graphics Processing Units (GPUs) and multi-core processors, and on the
OpenCL programming standard. It can be used on clusters as well as individual
hosts.

Documentation on configuring, building, and running the SHOC benchmark
programs is contained in the SHOC user manual, in the doc subdirectory
of the SHOC source code tree.  The file INSTALL.txt contains a sketch of
those instructions for rapid installation.

Installation should be familiar to anyone who is experienced with configure
and make, see the config directory for some examples.  Also, if your
platform requires regenerating the configure script, see build-aux/bootstrap.sh
and the manual for more details.

Last update: 2014-04-13 15:39:22 kspaff

