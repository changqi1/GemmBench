#!/bin/bash

g++ -std=c++11 ./cpu_sgemm_and_matmul.cpp -o ./cpu_sgemm_and_matmul -I../local/dnnl-v1.2.1/include/ -L../local/dnnl-v1.2.1/lib64/ -ldnnl -I../dnnl-v1.2.1/examples/
g++ -std=c++11 ./matmul.cpp -o matmul -I../local/dnnl-v1.2.1/include/ -L../local/dnnl-v1.2.1/lib64/ -ldnnl -I../dnnl-v1.2.1/examples/
