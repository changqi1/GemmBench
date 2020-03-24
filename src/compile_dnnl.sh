#!/bin/bash

MKLDNN_CFLAGS='../local/dnnl-v1.3/include'
MKLDNN_LFLAGS='../local/dnnl-v1.3/lib64'

source /opt/intel/mkl/bin/mklvars.sh intel64

ld_path=`pwd`/../
export LD_LIBRARY_PATH=$ld_path/local/dnnl-v1.3/lib64:$LD_LIBRARY_PATH

g++ -std=c++11 gemmbench_dnnl.cc -o gemmbench_dnnl -O2 -fopenmp -liomp5 -xCORE-AVX512 -lmkl_rt -I$MKLDNN_CFLAGS -L$MKLDNN_LFLAGS -ldnnl -I../local/eigen-v3.3.7/include/eigen3/ -I../local/xbyak-v5.891
