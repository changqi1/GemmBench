#!/bin/bash

# core_num=1
core_num=8
# core_num=16
# core_num=24

# cpus="24"
cpus="24-31"
# cpus="24-39"
# cpus="24-47"
# cpus="24-47,120-143"

####################################################################
####################### Do Not Modification ########################
####################################################################

echo ">> System Configuration"
lscpu | grep -E "Model name|NUMA|cache"
echo ">> Software Configuration"
echo "Core number: "$core_num
echo "cpus: "$cpus

cd src
sh ./compile.sh

export OMP_NUM_THREADS=$core_num
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

source /opt/intel/mkl/bin/mklvars.sh intel64

ld_path=`pwd`/../
export LD_LIBRARY_PATH=$ld_path/obj/local/dnnl-v1.5/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ld_path/obj/local/libxsmm-v1.16.1/lib/:$LD_LIBRARY_PATH

numactl -C $cpus -l ./gemmbench_dnnl $1 $2 $3
