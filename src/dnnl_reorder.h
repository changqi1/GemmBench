#ifndef __DNNL_REORDER__
#define __DNNL_REORDER__

#include "dnnl_common.h"

#include <typeinfo>

template <typename T_input, typename T_output>
bool Reorder(engine eng, stream stm, T_input* input, T_output* output, int size)
{
    std::stringstream weights_addr;
    weights_addr << "Reorder-" << size;

    memory::data_type src_dtype;
    memory::data_type dst_dtype;

    if (typeid(*input) == typeid(float)) {
        src_dtype = memory::data_type::f32;
	weights_addr << "-src_f32";
    }
    else if (typeid(*input) == typeid(bfloat16)) {
        src_dtype = memory::data_type::bf16;
	weights_addr << "-src_bf16";
    }
    else {
        std::cout << "src data type do not support" << std::endl;
        return false;
    }

    if (typeid(*output) == typeid(float)) {
        dst_dtype = memory::data_type::f32;
	weights_addr << "-dst_bf16";
    }
    else if (typeid(*output) == typeid(bfloat16)) {
        dst_dtype = memory::data_type::bf16;
	weights_addr << "-dst_bf16";
    }
    else {
        std::cout << "dst data type do not support" << std::endl;
        return false;
    }

    memory::dims dims = {size};
    auto src_md = memory::desc(dims, src_dtype, memory::format_tag::a);
    auto dst_md = memory::desc(dims, dst_dtype, memory::format_tag::a);
    auto src_mem = memory(src_md, eng, input);
    auto dst_mem = memory(dst_md, eng, output);

    std::string prim_key = weights_addr.str();
    auto it_prim_created = g_prim.find(prim_key);
    if (it_prim_created == g_prim.end()) {
        auto reorder_pd = reorder::primitive_desc(eng, src_md, eng, dst_md);
        auto *prim = new reorder(reorder_pd);
        g_prim.insert(std::pair<std::string, primitive *>(prim_key, prim));
        std::cout << "Reorder: save prim_key = " << prim_key << ", prim number = " << g_prim.size() << std::endl;

        prim->execute(stm, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, dst_mem}});
    }
    else {
        it_prim_created->second->execute(stm, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, dst_mem}});
    }
    stm.wait();

    return true;
}

#endif
