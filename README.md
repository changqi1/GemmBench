
# gemmbench

- `build_gemmbench.sh` to build mkldnn, eigen, xbyak env
- `./run.sh 128 768 768` to run gemm by kernel size `m n k`
- `vim run.sh` to config benchmark


benchmark list:

- mkl sgemm
- mkldnn sgemm
- mkldnn gemm_bf16bf16f32
- mkldnn gemm_bf16bf16f32 + openmp_convert
- ...
