
# gemmbench

- `build_gemmbench.sh` to build mkldnn, eigen, xbyak env
- `./run.sh 128 768 768` to run gemm by kernel size `m n k`
- `vim run.sh` to config benchmark


## output

```shell
$ ./run_dnnl.sh 128 768 768
./run_dnnl.sh 128 768 768
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
InnerProduct: save prim_key = InnerProduct-ffff-128-768-768-0x7f424f05e010, prim number = 1
result: 931.703,931.703
InnerProduct: save prim_key = InnerProduct2-ffff-128-768-768-0x7f424f05e010, prim number = 2
InnerProduct: reorder user_src_memory !!!
InnerProduct: reorder user_weights_memory !!!
InnerProduct: reorder user_bias_memory !!!
result: 933.069,931.703
InnerProduct: save prim_key = InnerProduct2-fffb-128-768-768-0x7f424f05e010, prim number = 3
InnerProduct: reorder user_src_memory !!!
InnerProduct: reorder user_weights_memory !!!
InnerProduct: reorder user_bias_memory !!!
result: 932,932
InnerProduct: save prim_key = InnerProduct2-fbbb-128-768-768-0x7f4252acf010, prim number = 4
InnerProduct: reorder user_src_memory !!!
result: 932,932
InnerProduct: save prim_key = InnerProduct-bbbb-128-768-768-0x7f4252acf010, prim number = 5
result: 932,932
InnerProduct: save prim_key = InnerProduct-bbbf-128-768-768-0x7f4252acf010, prim number = 6
result: 933.023,933.023
InnerProduct: save prim_key = InnerProductEltwise-bbbb-128-768-768-0x7f4252acf010, prim number = 7
result: 932,932

>> omp num_procs: 24
eigen sgemm:                    0.326602
mkl sgemm:                      0.111086 ms --> baseline
mkl sgemm+transB:               0.072265        +1.537X
dnnl sgemm:                     0.059063        +1.881X
dnnl bgemm:                     0.042495        +2.614X
dnnl bgemm+transB:              0.042278        +2.627X
dnnl bgemm+cvt:                 0.068613        +1.619X
dnnl bgemm+transB+cvt:          0.068234        +1.628X
dnnl bgemm+omp_cvt:             0.064976        +1.710X
dnnl bgemm+transB+omp_cvt:      0.065474        +1.697X
dnnl cvt f2b:                   0.007104        t/bgemm:   16.718%
dnnl omp_cvt f2b:               0.002415        t/bgemm:   5.683%
dnnl cvt b2f:                   0.008728        t/bgemm:   20.538%
dnnl omp_cvt b2f:               0.002441        t/bgemm:   5.744%
>> f: fp32, b: bf16, elw: eltwise
dnnl inner_product  ffff:       0.073340        +1.515X
dnnl inner_product2 ffff:       0.152852        +0.727X
dnnl inner_product2 fffb:       0.153917        +0.722X
dnnl inner_product2 fbbb:       0.155151        +0.716X
dnnl inner_product  bbbb:       0.043843        +2.534X
dnnl inner_product  bbbb+elw:   0.043843        +2.534X
dnnl inner_product  bbbf:       0.056434        +1.968X
```

## Tips

### inner product

```shell
>> src(N,IC) × weights(OC,IC) + bias(OC) = dst(N,OC)
>> 以上表示的是2维的 tensor，当输入为4维 tensor, src(N,IC′,IH,IW), weights(OC,IC′,KH,KW) 时，可以定义 IC=IC′*IH*IW，并且需要 KH=IH，KW=IW。
>> 只需要修改 memory::dims user memory::desc 的 format_tag
>> forward post-op 支持 eltwise
```

