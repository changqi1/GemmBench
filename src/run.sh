#!/bin/bash

# core_num=1
# core_num=8
# core_num=16
core_num=24
echo "Core number: "$core_num

# cpus="24"
# cpus="24-31"
# cpus="24-39"
cpus="24-47"
# cpus="24-47,120-143"
echo "cpus: "$cpus

####################################################################

sh ./compile.sh

export OMP_NUM_THREADS=$core_num
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

source /opt/intel/mkl/bin/mklvars.sh intel64

ld_path=`pwd`/../
export LD_LIBRARY_PATH=$ld_path/local/mkldnn-v1.0.4/lib64/:$LD_LIBRARY_PATH

numactl -C $cpus --membind=1 ./gemmbench $1 $2 $3
