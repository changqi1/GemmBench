#!/bin/bash

DNNL_CFLAGS='../obj/local/dnnl-v1.5/include'
DNNL_LFLAGS='../obj/local/dnnl-v1.5/lib64'

EIGEN_CFLAGS='../obj/local/eigen-v3.3.7/include/eigen3/'
XBYAK_CFLAGS='../obj/local/xbyak-v5.891'

LIBXSMM_CFLAGS='../obj/local/libxsmm-v1.16.1/include'
LIBXSMM_LFLAGS='../obj/local/libxsmm-v1.16.1/lib'

source /opt/intel/mkl/bin/mklvars.sh intel64
export LD_LIBRARY_PATH=`pwd`/../obj/local/dnnl-v1.5/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=`pwd`/../obj/local/libxsmm-v1.16.1/lib:$LD_LIBRARY_PATH

g++ -std=c++11 gemmbench_dnnl.cc dnnl_common.h dnnl_inner_product.h dnnl_matmul.h libxsmm_matmul.h matrix.h -o gemmbench_dnnl -O2 -fopenmp -liomp5 -xCORE-AVX512 -lmkl_rt -I$DNNL_CFLAGS -L$DNNL_LFLAGS -ldnnl -I$EIGEN_CFLAGS -I$XBYAK_CFLAGS -I$LIBXSMM_CFLAGS -L$LIBXSMM_LFLAGS -lxsmm -lblas -Ofast
# g++ -std=c++11 gemmbench_dnnl.cc dnnl_common.h dnnl_inner_product.h dnnl_matmul.h -o gemmbench_dnnl -O2 -fopenmp -liomp5 -I$DNNL_CFLAGS -L$DNNL_LFLAGS -ldnnl -I$XBYAK_CFLAGS -I$EIGEN_CFLAGS -DMKL_ILP64 -I${MKLROOT}/include -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

echo "compile successfully"
