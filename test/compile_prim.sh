#!/bin/bash

MKLDNN_CFLAGS='../local/dnnl-v1.3/include'
MKLDNN_LFLAGS='../local/dnnl-v1.3/lib64'

ld_path=`pwd`/../
export LD_LIBRARY_PATH=$ld_path/local/dnnl-v1.3/lib64:$LD_LIBRARY_PATH

g++ -std=c++11 dnnl_inner_product.h -o dnnl_inner_product -O2 -fopenmp -xCORE-AVX512 -I$MKLDNN_CFLAGS -L$MKLDNN_LFLAGS -ldnnl
