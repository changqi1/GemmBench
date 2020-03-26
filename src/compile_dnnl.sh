#!/bin/bash

DNNL_CFLAGS='../local/dnnl-v1.3/include'
DNNL_LFLAGS='../local/dnnl-v1.3/lib64'

source /opt/intel/mkl/bin/mklvars.sh intel64

ld_path=`pwd`/../
export LD_LIBRARY_PATH=$ld_path/local/dnnl-v1.3/lib64:$LD_LIBRARY_PATH

EIGEN_CFLAGS='../local/eigen-v3.3.7/include/eigen3/'
XBYAK_CFLAGS='../local/xbyak-v5.891'

g++ -std=c++11 gemmbench_dnnl.cc dnnl_common.h dnnl_inner_product.h dnnl_matmul.h -o gemmbench_dnnl -O2 -fopenmp -liomp5 -xCORE-AVX512 -lmkl_rt -I$DNNL_CFLAGS -L$DNNL_LFLAGS -ldnnl -I$EIGEN_CFLAGS -I$XBYAK_CFLAGS
