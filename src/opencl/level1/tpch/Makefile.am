include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(OCL_LDFLAGS)
CPPFLAGS += $(OCL_CPPFLAGS)
L1LIBS = $(OCL_LIBS) $(LIBS)

# what is the destination for programs built from this directory?
serialopencldir = $(bindir)/Serial/OpenCL

# what programs should be installed in the bin/Serial/OpenCL destination?
serialopencl_PROGRAMS = Tpch

#Currently innerJoin, project, select, and unique are included 
Tpch_SOURCES = innerJoin.cpp project.cpp select.cpp unique.cpp tpch.cpp main.cpp
Tpch_LDADD = $(L1LIBS)
