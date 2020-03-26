
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
InnerProduct: save prim_key = InnerProduct-ffff-128-768-768-0x7f9e628eb010, prim number = 1
result: 930.378,930.378
result: 930.378,930.378
InnerProduct: save prim_key = InnerProduct-fffb-128-768-768-0x7f9e628eb010, prim number = 2
InnerProduct: reorder user_src_memory !!!
InnerProduct: reorder user_weights_memory !!!
InnerProduct: reorder user_bias_memory !!!
result: 932,932
InnerProduct: save prim_key = InnerProduct-fbbb-128-768-768-0x7f9e6635b010, prim number = 3
InnerProduct: reorder user_src_memory !!!
result: 932,932
InnerProduct: save prim_key = InnerProduct-bbbb-128-768-768-0x7f9e6635b010, prim number = 4
result: 932,932
InnerProduct: save prim_key = InnerProduct-bbbf-128-768-768-0x7f9e6635b010, prim number = 5
result: 933.023,933.023

>> omp num_procs: 24
eigen sgemm:                    0.322845
mkl sgemm:                      0.111739 ms --> baseline
mkl sgemm+transB:               0.072397        +1.543X
dnnl sgemm:                     0.059460        +1.879X
dnnl bgemm:                     0.042808        +2.610X
dnnl bgemm+transB:              0.042398        +2.635X
dnnl bgemm+cvt:                 0.068947        +1.621X
dnnl bgemm+transB+cvt:          0.067899        +1.646X
dnnl bgemm+omp_cvt:             0.063947        +1.747X
dnnl bgemm+transB+omp_cvt:      0.065532        +1.705X
dnnl cvt:                       0.007120        t/bgemm:   16.631%
dnnl omp_cvt:                   0.002453        t/bgemm:   5.730%
>> f: fp32, b: bf16
dnnl inner_product  ffff:       0.073177        +1.527X
dnnl inner_product2 ffff:       0.099385        +1.124X
dnnl inner_product2 fffb:       0.156444        +0.714X
dnnl inner_product2 fbbb:       0.154767        +0.722X
dnnl inner_product  bbbb:       0.043306        +2.580X
dnnl inner_product  bbbf:       0.057032        +1.959X
```
