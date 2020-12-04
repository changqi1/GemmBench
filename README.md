
# gemmbench

- `build.sh` to build mkldnn, eigen, xbyak env
- `vim run.sh` to config benchmark
- `./run.sh 128 768 768` to run gemm by kernel size `m n k`


## output

```shell
$ ./run.sh 128 1024 1024
>> System Configuration
NUMA node(s):        4
Model name:          Intel(R) Xeon(R) Platinum 8369HC CPU @ 3.40GHz
L1d cache:           32K
L1i cache:           32K
L2 cache:            1024K
L3 cache:            33792K
NUMA node0 CPU(s):   0-23,96-119
NUMA node1 CPU(s):   24-47,120-143
NUMA node2 CPU(s):   48-71,144-167
NUMA node3 CPU(s):   72-95,168-191
>> Software Configuration
Core number: 8
cpus: 24-31
compile successfully
m = 128, n = 1024, k = 1024
A(128, 1024) * B(1024, 1024) = C(128, 1024)
A padding -> 1028
B padding -> 1028
C padding -> 1028

starting...
result: 1239.04,1239.04
result: 1239.04,1239.04
result: 1239.04,1239.04
result: 1239.04,1239.04
result: 1239.04,1239.04
result: 1239.04,1239.04
result: 1242.56,1242.56
result: 1242.56,1242.56
result: 1242.56,1242.56
result: 1242.56,1242.56
result: 1242.56,1242.56
result: 1242.56,1242.56
result: 1242.56,1242.56
result: 1242.56,1242.56
result: 1242.56,1242.56
result: 1242.56,1242.56
InnerProduct: save prim_key = InnerProduct-ffff-128-1024-1024-0x7fbec9b2b010, prim number = 1
result: 1241.9,1241.9
InnerProduct: save prim_key = InnerProduct2-ffff-128-1024-1024-0x7fbec9b2b010, prim number = 2
InnerProduct: reorder user_src_memory !!!
InnerProduct: reorder user_weights_memory !!!
InnerProduct: reorder user_bias_memory !!!
result: 1242.14,1241.9
InnerProduct: save prim_key = InnerProduct2-fffb-128-1024-1024-0x7fbec9b2b010, prim number = 3
InnerProduct: reorder user_src_memory !!!
InnerProduct: reorder user_weights_memory !!!
InnerProduct: reorder user_bias_memory !!!
result: 1240,1240
InnerProduct: save prim_key = InnerProduct2-fbbb-128-1024-1024-0x7fbec9fad010, prim number = 4
InnerProduct: reorder user_src_memory !!!
result: 1240,1240
InnerProduct: save prim_key = InnerProduct-bbbb-128-1024-1024-0x7fbec9fad010, prim number = 5
result: 1240,1240
InnerProduct: save prim_key = InnerProductEltwise-bbbb-128-1024-1024-0x7fbec9fad010, prim number = 6
result: 1240,1240
InnerProduct: save prim_key = InnerProduct-bbbf-128-1024-1024-0x7fbec9fad010, prim number = 7
result: 1243.66,1243.66
InnerProduct: save prim_key = InnerProduct-bbff-128-1024-1024-0x7fbec9fad010, prim number = 8
result: 1243.66,1243.66
MatMul: save prim_key = MatMul-ffff-128-1024-1024-0x7fbec9b2b010, prim number = 9
result: 1241.9,1241.9
MatMul: save prim_key = MatMul-bbbb-128-1024-1024-0x7fbec9fad010, prim number = 10
result: 1240,1240
MatMul: save prim_key = MatMul-bbbf-128-1024-1024-0x7fbec9fad010, prim number = 11
result: 1243.66,1243.66
MatMul2: save prim_key = MatMul2-fff-128-1024-1024-0x7fbec9b2b010, prim number = 12
result: 1240.8,1240.8
MatMul2: save prim_key = MatMul2-bbb-128-1024-1024-0x7fbec9fad010, prim number = 13
result: 1240,1240
result: 1240,1240
MatMul2: save prim_key = MatMul2-bbf-128-1024-1024-0x7fbec9fad010, prim number = 14
result: 1242.56,1242.56
BatchMatMul: save prim_key = BatchMatMul-bbb-128-1024-1024-0x7fbe9697e010, prim number = 15
result: 1240,1240

>> omp num_procs: 8
eigen sgemm:                    0.288479
mkl sgemm:                      0.239387 ms --> baseline
mkl sgemm+pad:                  0.213381        +1.122X
mkl sgemm+transB:               0.253224        +0.945X
mkl sgemm+transB+pad:           0.214379        +1.117X
dnnl sgemm:                     0.471490        +0.508X
dnnl bgemm:                     0.127462        +1.878X
dnnl bgemm+transB:              0.117585        +2.036X
dnnl bgemm+cvt:                 0.149490        +1.601X
dnnl bgemm+transB+cvt:          0.150828        +1.587X
dnnl bgemm+omp_cvt:             0.137500        +1.741X
dnnl bgemm+transB+omp_cvt:      0.129826        +1.844X
dnnl cvt f2b:                   0.009332        t/bgemm:   7.321%
dnnl omp_cvt f2b:               0.002464        t/bgemm:   1.933%
dnnl cvt b2f:                   0.011268        t/bgemm:   8.840%
dnnl omp_cvt b2f:               0.002471        t/bgemm:   1.938%
>> inner_product, f: fp32, b: bf16, elw: eltwise
dnnl inner_product  ffff:       0.487376        +0.491X
dnnl inner_product2 ffff:       0.150314        +1.593X
dnnl inner_product2 fffb:       0.147939        +1.618X
dnnl inner_product2 fbbb:       0.149016        +1.606X
dnnl inner_product  bbbb:       0.126896        +1.886X
dnnl inner_product  bbbb+elw:   0.127095        +1.884X
dnnl inner_product  bbbf:       0.143308        +1.670X
dnnl inner_product  bbff:       0.143234        +1.671X
>> matmul, f: fp32, b: bf16, elw: eltwise
dnnl matmul ffff:               0.487054        +0.491X
dnnl matmul bbbb:               0.126415        +1.894X
dnnl matmul bbbf:               0.140714        +1.701X
dnnl matmul2 fff:               0.475203        +0.504X
dnnl matmul2 bbb:               0.127512        +1.877X
dnnl matmul2 bbb+elw:           0.125903        +1.901X
dnnl matmul2 bbf:               0.131852        +1.816X
dnnl 10 batch matmul bbb:       0.189672        +1.262X
```

## Tips

```shell
>> inner product
src(N,IC) × weights(OC,IC) + bias(OC) = dst(N,OC)
以上表示的是2维的 tensor，当输入为4维 tensor, src(N,IC′,IH,IW), weights(OC,IC′,KH,KW) 时，
可以定义 IC=IC′*IH*IW，并且需要 KH=IH，KW=IW,
只需要修改 memory::dims user memory::desc 的 format_tag。
forward post-op 支持 eltwise

>> inner_product 内部engine依据当前参数类型配置而定
>> inner_product2 内部engine采用bf16硬件进行计算

>> matmul 有BiasAdd的操作
>> matmul2 没有BiasAdd的操作
```

