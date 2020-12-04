#!/bin/bash

# need to install intel mkl
sudo dnf --enablerepo=PowerTools install blas-devel
cur_path=`pwd`

# build mkl-dnn-v1.0.4
#cd $cur_path/obj
#git clone https://github.com/intel/mkl-dnn.git mkldnn-v1.0.4
#cd mkldnn-v1.0.4/
#git checkout v1.0.4
#git apply --check ../patch_mkldnn-v1.0.4_bgemm.diff
#git apply ../patch_mkldnn-v1.0.4_bgemm.diff
#mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../../local/mkldnn-v1.0.4/ ..
#make -j all
#make install

# build dnnl-v1.1.3
#cd $cur_path/obj
#git clone https://github.com/intel/mkl-dnn.git dnnl-v1.1.3
#cd dnnl-v1.1.3/
#git checkout v1.1.3
#git apply --check ../patch_dnnl-v1.1.3_bgemm.diff
#git apply ../patch_dnnl-v1.1.3_bgemm.diff
#mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../../local/dnnl-v1.1.3/ ..
#make -j all
#make install

# build dnnl-v1.2.1
#cd $cur_path/obj
#git clone https://github.com/intel/mkl-dnn.git dnnl-v1.2.1
#cd dnnl-v1.2.1/
#git checkout v1.2.1
#git apply --check ../patch_dnnl-v1.2.1_bgemm.diff
#git apply ../patch_dnnl-v1.2.1_bgemm.diff
#mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../../local/dnnl-v1.2.1/ ..
#make -j all
#make install

# build dnnl-v1.3
#cd $cur_path/obj
#git clone https://github.com/intel/mkl-dnn.git dnnl-v1.3
#cd dnnl-v1.3/
#git checkout v1.3
#git apply --check ../patch_dnnl-v1.3_bgemm.diff
#git apply ../patch_dnnl-v1.3_bgemm.diff
#mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../../local/dnnl-v1.3/ ..
#make -j all
#make install

# build dnnl-v1.4
#cd $cur_path/obj
#git clone https://github.com/intel/mkl-dnn.git dnnl-v1.4
#cd dnnl-v1.4/
#git checkout v1.4
#git apply --check ../patch_dnnl-v1.4_bgemm.diff
#git apply ../patch_dnnl-v1.4_bgemm.diff
#mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../../local/dnnl-v1.4/ ..
#make -j all
#make install

# build dnnl-v1.5
cd $cur_path/obj
git clone https://github.com/intel/mkl-dnn.git dnnl-v1.5
cd dnnl-v1.5/
git checkout v1.5
git apply --check ../patch_dnnl-v1.5_bgemm.diff
git apply --check ../patch_dnnl-v1.5_boost_small_kernel_sgemm.diff
git apply ../patch_dnnl-v1.5_bgemm.diff
git apply ../patch_dnnl-v1.5_boost_small_kernel_sgemm.diff
mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../../local/dnnl-v1.5/ ..
make -j all
make install

# build xbyak
#cd $cur_path/obj
#git clone https://github.com/herumi/xbyak.git xbyak-v5.891
#cd xbyak-v5.891
#git checkout v5.891
#mkdir -p ../local/xbyak-v5.891/
#cp -pR xbyak/*.h ../local/xbyak-v5.891/

# build eigen
cd $cur_path/obj
git clone https://github.com/eigenteam/eigen-git-mirror.git eigen-v3.3.7
cd eigen-v3.3.7
git checkout 3.3.7
mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../../local/eigen-v3.3.7/ ..
make install

# build libxsmm
cd $cur_path/obj
git clone https://github.com/hfp/libxsmm.git libxsmm-v1.16.1
cd libxsmm-v1.16.1
git checkout 1.16.1
make PREFIX=../local/libxsmm-v1.16.1/ STATIC=0 install -j8
