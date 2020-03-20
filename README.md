
# gemmbench

- `build_gemmbench.sh` to build mkldnn, eigen, xbyak env
- `cd src`
- `./run.sh 128 768 768` to run gemm by kernel size `m n k`


benchmark list:

- mkl sgemm
- mkldnn sgemm
- mkldnn gemm_bf16bf16f32
- mkldnn gemm_bf16bf16f32 + openmp_convert
- ...