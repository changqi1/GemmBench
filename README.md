
# gemmbench

- `build_gemmbench.sh` to build mkldnn, eigen, xbyak env
- `./run.sh 128 768 768` to run gemm by kernel size `m n k`
- `vim run.sh` to config benchmark


benchmark list:

```shell
# ./run_dnnl.sh 128 768 768
Core number: 24
cpus: 24-47
./gemmbench_dnnl m n k
argc = 4
argv[0] --> ./gemmbench_dnnl
argv[1] --> 128
argv[2] --> 768
argv[3] --> 768

starting...
result: 929.279,929.279
result: 929.278,929.278
result: 929.282,929.282
result: 929.278,929.278
result: 931.922,931.922
result: 931.922,931.922
result: 931.922,931.922
result: 931.922,931.922
result: 931.922,931.922
result: 931.922,931.922
result: 931.922,931.922
result: 931.922,931.922
result: 931.922,931.922
result: 931.922,931.922
InnerProduct: save prim_key = InnerProduct-ffff-128-768-768-0x7f50b93f5010, prim number = 1
result: 931.703,931.703
result: 931.703,931.703
InnerProduct: save prim_key = InnerProduct-fffb-128-768-768-0x7f50b93f5010, prim number = 2
InnerProduct: reorder user_src_memory !!!
InnerProduct: reorder user_weights_memory !!!
InnerProduct: reorder user_bias_memory !!!
result: 932,932
InnerProduct: save prim_key = InnerProduct-fbbb-128-768-768-0x7f50bce65010, prim number = 3
InnerProduct: reorder user_src_memory !!!
result: 932,932
InnerProduct: save prim_key = InnerProduct-bbbb-128-768-768-0x7f50bce65010, prim number = 4
result: 932,932
InnerProduct: save prim_key = InnerProduct-bbbf-128-768-768-0x7f50bce65010, prim number = 5
result: 933.023,933.023

>> omp num_procs: 24
eigen sgemm:                    0.332314
mkl sgemm:                      0.114996 ms --> baseline
mkl sgemm+transB:               0.073181        +1.571X
dnnl sgemm:                     0.058244        +1.974X
dnnl bgemm:                     0.043647        +2.635X
dnnl bgemm+transB:              0.042602        +2.699X
dnnl bgemm+cvt:                 0.069673        +1.651X
dnnl bgemm+transB+cvt:          0.067952        +1.692X
dnnl bgemm+omp_cvt:             0.064745        +1.776X
dnnl bgemm+transB+omp_cvt:      0.064746        +1.776X
dnnl cvt f2b:                   0.007179        t/bgemm:   16.448%
dnnl omp_cvt f2b:               0.002603        t/bgemm:   5.963%
dnnl cvt b2f:                   0.008772        t/bgemm:   20.098%
dnnl omp_cvt b2f:               0.002403        t/bgemm:   5.505%
>> f: fp32, b: bf16
dnnl inner_product  ffff:       0.073543        +1.564X
dnnl inner_product2 ffff:       0.073775        +1.559X
dnnl inner_product2 fffb:       0.156718        +0.734X
dnnl inner_product2 fbbb:       0.156653        +0.734X
dnnl inner_product  bbbb:       0.046033        +2.498X
dnnl inner_product  bbbf:       0.058333        +1.971X
```
