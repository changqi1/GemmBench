#include "mkldnn.hpp"
#include "bfloat16.hpp"
#include "mkldnn_debug.h"

#include <iostream>
#include <iomanip>
#include <chrono>

#include <mkl.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace mkldnn;


int main(int argc, char *argv[])
{
    printf("argc = %d\n", argc);
    for(int ndx = 0; ndx != argc; ++ndx)
        printf("argv[%d] --> %s\n", ndx,argv[ndx]);

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    mkldnn::impl::bfloat16_t *A_bf16 = new mkldnn::impl::bfloat16_t[m*k];
    mkldnn::impl::bfloat16_t *B_bf16 = new mkldnn::impl::bfloat16_t[k*n];
    float *A = new float[m*k];
    float *B = new float[k*n];
    float *C = new float[m*n];
    //float C[m*n] = {0};

    for (int i = 0; i < m*k; ++i) {
	A[i] = 1.1;
	A_bf16[i] = (mkldnn::impl::bfloat16_t)1.1;
    }

    for (int i = 0; i < k*n; ++i) {
	B[i] = 1.1;
	B_bf16[i] = (mkldnn::impl::bfloat16_t)1.1;
    }

    for (int i = 0; i < m*n; ++i) {
	C[i] = 0.1;
    }

    MatrixXf A_mat(m, k);
    MatrixXf B_mat(k, n);
    MatrixXf C_mat(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            A_mat.row(i).col(j) << 1.1;
        }
    }
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            B_mat.row(i).col(j) << 1.1;
        }
    }

    std::cout << "starting..." << std::endl;
    auto tag_1 = std::chrono::high_resolution_clock::now();
    auto tag_2 = std::chrono::high_resolution_clock::now();

/*
    C_mat = A_mat * B_mat;
    tag_1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        C_mat = A_mat * B_mat;
    }
    tag_2 = std::chrono::high_resolution_clock::now();
    auto tag_d1 = std::chrono::duration<double>(tag_2 - tag_1).count();
    std::cout << "eigen_matmul() time: " << tag_d1 << " ms" << std::endl;
    std::cout << "result: " << C_mat(0, 0) << "," << C_mat(m-1, n-1) << std::endl;
*/


    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    tag_1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    }
    tag_2 = std::chrono::high_resolution_clock::now();
    auto tag_d2 = std::chrono::duration<double>(tag_2 - tag_1).count();
    std::cout << "cblas_sgemm() time: " << tag_d2 << " ms" << std::endl;
    std::cout << "result: " << C[0] << "," << C[m*n-1] << std::endl;


    dnnl_gemm_bf16bf16f32('N', 'N', m, n, k, 1.0, A_bf16, k, B_bf16, n, 0.0, C, n);
    tag_1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
	dnnl_gemm_bf16bf16f32('N', 'N', m, n, k, 1.0, A_bf16, k, B_bf16, n, 0.0, C, n);
    }
    tag_2 = std::chrono::high_resolution_clock::now();
    auto tag_d3 = std::chrono::duration<double>(tag_2 - tag_1).count();
    std::cout << "mkldnn_gemm_bf16bf16f32() time: " << tag_d3 << " ms      \t+ " << float(tag_d2/tag_d3)*100 << " %" << std::endl;
    std::cout << "result: " << C[0] << "," << C[m*n-1] << std::endl;

    dnnl::impl::cvt_float_to_bfloat16((dnnl::impl::bfloat16_t *)A_bf16, A, m*k);
    dnnl_gemm_bf16bf16f32('N', 'N', m, n, k, 1.0, A_bf16, k, B_bf16, n, 0.0, C, n);
    tag_1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        dnnl::impl::cvt_float_to_bfloat16((dnnl::impl::bfloat16_t *)A_bf16, A, m*k);
	dnnl_gemm_bf16bf16f32('N', 'N', m, n, k, 1.0, A_bf16, k, B_bf16, n, 0.0, C, n);
    }
    tag_2 = std::chrono::high_resolution_clock::now();
    auto tag_d4 = std::chrono::duration<double>(tag_2 - tag_1).count();
    std::cout << "mkldnn_gemm_bf16bf16f32() + cvt time: " << tag_d4 << " ms  \t+ " << float(tag_d2/tag_d4)*100 << " %" << std::endl;
    std::cout << "result: " << C[0] << "," << C[m*n-1] << std::endl;

    dnnl::impl::cvt_float_to_bfloat16((dnnl::impl::bfloat16_t *)A_bf16, A, m*k);
    tag_1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        dnnl::impl::cvt_float_to_bfloat16((dnnl::impl::bfloat16_t *)A_bf16, A, m*k);
    }
    tag_2 = std::chrono::high_resolution_clock::now();
    auto tag_d5 = std::chrono::duration<double>(tag_2 - tag_1).count();
    std::cout << "cvt time: " << tag_d5 << " ms  \t+ " << float(tag_d2/tag_d5)*100 << " %" << std::endl;
    std::cout << "result: " << C[0] << "," << C[m*n-1] << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
