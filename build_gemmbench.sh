#!/bin/bash

# build mkl-dnn-v1.0.4
git clone https://github.com/intel/mkl-dnn.git mkldnn-v1.0.4
cd mkldnn-v1.0.4/
git checkout v1.0.4
git apply --check ../patch_mkldnn-v1.0.4_bf16_structure_API_and_gemm_bf16bf16fp32_20200115.diff
git apply ../patch_mkldnn-v1.0.4_bf16_structure_API_and_gemm_bf16bf16fp32_20200115.diff
mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../../local/mkldnn-v1.0.4/ ..
make -j all
make install
cd ../../

# build dnnl-v1.1.3
git clone https://github.com/intel/mkl-dnn.git dnnl-v1.1.3
cd dnnl-v1.1.3/
git checkout v1.1.3
git apply --check ../patch_dnnl-v1.1.3.diff
git apply ../patch_dnnl-v1.1.3.diff
mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../../local/dnnl-v1.1.3/ ..
make -j all
make install
cd ../../

# build dnnl-v1.2.1
git clone https://github.com/intel/mkl-dnn.git dnnl-v1.2.1
cd dnnl-v1.2.1/
git checkout v1.2.1
git apply --check ../patch_dnnl-v1.2.1.diff
git apply ../patch_dnnl-v1.2.1.diff
mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../../local/dnnl-v1.2.1/ ..
make -j all
make install
cd ../../

# build dnnl-v1.3 for CPX
git clone https://github.com/intel/mkl-dnn.git dnnl-v1.3
cd dnnl-v1.3/
git checkout remotes/origin/rls-v1.3
git apply --check ../patch_dnnl-v1.3.diff
git apply ../patch_dnnl-v1.3.diff
mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../../local/dnnl-v1.3/ ..
make -j all
make install
cd ../..

# build xbyak
git clone https://github.com/herumi/xbyak.git xbyak-v5.891
cd xbyak-v5.891
git checkout v5.891
mkdir -p ../local/xbyak-v5.891/
cp -pR xbyak/*.h ../local/xbyak-v5.891/
cd ../

# build eigen
git clone https://github.com/eigenteam/eigen-git-mirror.git eigen-v3.3.7
cd eigen-v3.3.7
git checkout 3.3.7
mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../../local/eigen-v3.3.7/ ..
make install
cd ../../

