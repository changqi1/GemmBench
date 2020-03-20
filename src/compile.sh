#!/bin/bash

MKLDNN_CFLAGS='../local/mkldnn-v1.0.4/include'
MKLDNN_LFLAGS='../local/mkldnn-v1.0.4/lib64'

source /opt/intel/mkl/bin/mklvars.sh intel64

ld_path=`pwd`/../
export LD_LIBRARY_PATH=$ld_path/local/mkldnn-v1.0.4/lib64:$LD_LIBRARY_PATH

g++ -std=c++11 gemmbench.cc -o gemmbench -O2 -fopenmp -liomp5 -xCORE-AVX512 -lmkl_rt -I$MKLDNN_CFLAGS -L$MKLDNN_LFLAGS -lmkldnn -I../local/eigen-v3.3.7/include/eigen3/ -I../local/xbyak-v5.891
