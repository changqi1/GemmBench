//#ifndef __DNNL_INNER_PRODUCT__
//#define __DNNL_INNER_PRODUCT__

#include "dnnl.hpp"
#include "dnnl_debug.h"
#include "bfloat16.hpp"

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>

using namespace dnnl;

typedef dnnl::impl::bfloat16_t bfloat16;

typedef std::unordered_map<std::string, memory*> map_mem_t;
typedef std::unordered_map<std::string, inner_product_forward::primitive_desc*> map_primd_t;
typedef std::unordered_map<std::string, primitive*> map_prim_t;

/*
void InnerProduct(float* input, float* weight, float* output, int m, int n, int k);
void InnerProduct(float* input, float* weight, bfloat16* output, int m, int n, int k);
void InnerProduct(float* input, bfloat16* weight, bfloat16* output, int m, int n, int k);
void InnerProduct(bfloat16* input, float* weight, float* output, int m, int n, int k);
void InnerProduct(bfloat16* input, bfloat16* weight, float* output, int m, int n, int k);
void InnerProduct(bfloat16* input, bfloat16* weight, bfloat16* output, int m, int n, int k);

void InnerProduct(float* input, float* weight, float* bias, float* output, int m, int n, int k);
void InnerProduct(float* input, float* weight, float* bias, bfloat16* output, int m, int n, int k);
void InnerProduct(float* input, bfloat16* weight, bfloat16* bias, bfloat16* output, int m, int n, int k);
void InnerProduct(bfloat16* input, float* weight, float* bias, float* output, int m, int n, int k);
void InnerProduct(bfloat16* input, bfloat16* weight, bfloat16* bias, float* output, int m, int n, int k);
void InnerProduct(bfloat16* input, bfloat16* weight, bfloat16* bias, bfloat16* output, int m, int n, int k);
*/

bool InnerProduct(engine eng, stream stm, bfloat16* input, bfloat16* weight, bfloat16* bias, bfloat16* output, int m, int n, int k);

engine eng(engine::kind::cpu, 0);
map_mem_t g_memory;
map_primd_t g_prim_desc;
map_prim_t g_prim;

int main(void)
{
/*
    engine cpu_engine;
    stream cpu_stream;

    stream stream(eng);
    cpu_engine = eng;
    cpu_stream = stream;
*/

    engine cpu_engine(eng);
    stream cpu_stream(cpu_engine);
    int m = 128;
    int n = 768;
    int k = 768;

    bfloat16 *A_bf16 = new bfloat16[m*k];
    bfloat16 *B_bf16 = new bfloat16[k*n];
    bfloat16 *C_bf16 = new bfloat16[m*n];
    bfloat16 *D_bf16 = new bfloat16[n];

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            A_bf16[i*k+j] = (bfloat16)1.1;
        }
    }

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            B_bf16[i*n+j] = (bfloat16)1.1;
        }
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C_bf16[i*n+j] = (bfloat16)1.1;
        }
    }

    for (int i = 0; i < n; ++i) {
        D_bf16[i] = (bfloat16)1.1;
    }

    InnerProduct(cpu_engine, cpu_stream, A_bf16, B_bf16, D_bf16, C_bf16, m, n, k);
    auto tag_1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
        InnerProduct(cpu_engine, cpu_stream, A_bf16, B_bf16, D_bf16, C_bf16, m, n, k);
    auto tag_2 = std::chrono::high_resolution_clock::now();
    auto tag_diff = std::chrono::duration<double>(tag_2 - tag_1).count();
    std::cout << "tag_diff: " << tag_diff << std::endl;

    /*float *data = static_cast<float *>(dst_memory.get_data_handle());
    int size = dst_memory.get_desc().get_size() / sizeof(float);
    std::cout << size << std::endl;
    for (int k = 0; k < size; ++k) {
        printf("%f ", data[k]);
        if ((k + 1) % 6 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;*/

    for (map_primd_t::iterator iter = g_prim_desc.begin(); iter != g_prim_desc.end(); ++iter) {
      delete iter->second;
    }
    for (map_prim_t::iterator iter = g_prim.begin(); iter != g_prim.end(); ++iter) {
      delete iter->second;
    }
    for (map_mem_t::iterator iter = g_memory.begin(); iter != g_memory.end(); ++iter) {
      delete iter->second;
    }

    return 0;
}

bool InnerProduct(engine eng, stream stm, bfloat16* input, bfloat16* weight, bfloat16* bias, bfloat16* output, int m, int n, int k)
{
    const void *address = static_cast<const void*>(weight);

    std::stringstream weights_addr;
    weights_addr << m << "-";
    weights_addr << k << "-";
    weights_addr << n << "-";
    weights_addr << address;
    std::string prim_key = weights_addr.str();

    memory::dims src_tz = { m, k };
    memory::dims weights_tz = { n, k };
    memory::dims bias_tz = { n };
    memory::dims dst_tz = { m, n };

    auto it_prim_created = g_prim.find(prim_key);
    if (it_prim_created == g_prim.end()) {
        std::cout << "InnerProduct: m,n,k -> " << m << "," << n << "," << k << std::endl;
    
        auto src_md     = memory::desc({ src_tz }, memory::data_type::bf16, memory::format_tag::any);
        auto weights_md = memory::desc({ weights_tz }, memory::data_type::bf16, memory::format_tag::any);
        auto bias_md    = memory::desc({ bias_tz }, memory::data_type::bf16, memory::format_tag::any);
        auto dst_md     = memory::desc({ dst_tz }, memory::data_type::bf16, memory::format_tag::any);
        
        auto desc = inner_product_forward::desc(prop_kind::forward_inference, src_md, weights_md, bias_md, dst_md);

        auto *prim_desc = new inner_product_forward::primitive_desc(desc, eng);

        auto *prim = new inner_product_forward(*prim_desc);

        g_prim.insert(std::pair<std::string, primitive *>(prim_key, prim));
        g_prim_desc.insert(std::pair<std::string, inner_product_forward::primitive_desc *>(prim_key, prim_desc));
        std::cout << "InnerProduct: save prim_key -> " << prim_key << ", prim number -> " << g_prim.size() << std::endl;
    }

    //std::vector<bfloat16> bias(product(bias_tz), bfloat16(0.0));
    //std::vector<float> bias(product(bias_tz), 0.0);

    //memory::data_type dt = (std::is_floating_point<T>::value) ? memory::data_type::f32 : memory::data_type::bf16;

    auto user_src_md = memory::desc(src_tz, memory::data_type::bf16, memory::format_tag::nc);
    auto user_weights_md = memory::desc(weights_tz, memory::data_type::bf16, memory::format_tag::io); // cn or io
    auto user_bias_md = memory::desc(bias_tz, memory::data_type::bf16, memory::format_tag::x);

    auto user_src_memory = memory(user_src_md, eng, input);
    auto user_weights_memory = memory(user_weights_md, eng, weight);
    auto user_bias_memory = memory(user_bias_md, eng, bias);

    auto it_prim_desc_created = g_prim_desc.find(prim_key);
    if (it_prim_desc_created == g_prim_desc.end()) {
        std::cout << "InnerProduct error: can find g_prim_desc -> " << prim_key << std::endl;
        return false;
    }
    inner_product_forward::primitive_desc prim_desc = *it_prim_desc_created->second;

    auto src_memory = user_src_memory;
    auto weights_memory = &user_weights_memory;
    auto bias_memory = &user_bias_memory;
    auto dst_memory = memory(prim_desc.dst_desc(), eng, output);

    if (prim_desc.src_desc() != user_src_memory.get_desc()) {
        std::cout << "InnerProduct: reorder user_src_memory" << std::endl;
        src_memory = memory(prim_desc.src_desc(), eng);
        auto reorder_src = reorder(user_src_memory, src_memory);
        reorder_src.execute(stm, { { DNNL_ARG_FROM, user_src_memory },
                                          { DNNL_ARG_TO, src_memory } });
    }

    if (prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        std::string prim_weights_key = prim_key+"-weights";
        auto it_memory_created = g_memory.find(prim_weights_key);
        if (it_memory_created == g_memory.end()) {
            std::cout << "InnerProduct: reorder user_weights_memory" << std::endl;
            weights_memory = new memory(prim_desc.weights_desc(), eng);
            auto reorder_weights = reorder(user_weights_memory, *weights_memory);
            reorder_weights.execute(stm, {
                { DNNL_ARG_FROM, user_weights_memory },
                { DNNL_ARG_TO, *weights_memory } });
            g_memory.insert(std::pair<std::string, memory *>(prim_weights_key, weights_memory));
        }
        else {
            weights_memory = it_memory_created->second;
        }
    }

    if (prim_desc.bias_desc() != user_bias_memory.get_desc()) {
        std::string prim_bias_key = prim_key+"-bias";
        auto it_memory_created = g_memory.find(prim_bias_key);
        if (it_memory_created == g_memory.end()) {
            std::cout << "InnerProduct: reorder user_bias_memory" << std::endl;
            bias_memory = new memory(prim_desc.bias_desc(), eng);
            auto reorder_bias = reorder(user_bias_memory, *bias_memory);
            reorder_bias.execute(stm, {
                { DNNL_ARG_FROM, user_bias_memory },
                { DNNL_ARG_TO, *bias_memory } });
            g_memory.insert(std::pair<std::string, memory *>(prim_bias_key, bias_memory));
        }
        else {
            weights_memory = it_memory_created->second;
        }
    }

    it_prim_created = g_prim.find(prim_key);
    if (it_prim_created != g_prim.end()) {
        it_prim_created->second->execute(stm, {
            { DNNL_ARG_SRC, src_memory },
            { DNNL_ARG_WEIGHTS, *weights_memory },
            { DNNL_ARG_BIAS, *bias_memory },
            { DNNL_ARG_DST, dst_memory } });
    }
    else {
        std::cout << "InnerProduct: execute error, prim_key -> " << prim_key << std::endl;
        return false;
    }
    stm.wait();
}

memory::dim product(const memory::dims &dims)
{
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
           std::multiplies<memory::dim>());
}

//#endif
