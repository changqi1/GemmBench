#ifndef __DNNL_COMMON__
#define __DNNL_COMMON__

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
typedef std::unordered_map<std::string, inner_product_forward::primitive_desc*> map_ip_primd_t;
typedef std::unordered_map<std::string, matmul::primitive_desc*> map_mm_primd_t;
typedef std::unordered_map<std::string, primitive*> map_prim_t;

engine eng(engine::kind::cpu, 0);

map_mem_t g_memory;
map_ip_primd_t g_ip_prim_desc;
map_mm_primd_t g_mm_prim_desc;
map_prim_t g_prim;

void del_dnnl(void)
{
    for (map_mem_t::iterator iter = g_memory.begin(); iter != g_memory.end(); ++iter) {
      delete iter->second;
    }

    for (map_ip_primd_t::iterator iter = g_ip_prim_desc.begin(); iter != g_ip_prim_desc.end(); ++iter) {
      delete iter->second;
    }

    for (map_mm_primd_t::iterator iter = g_mm_prim_desc.begin(); iter != g_mm_prim_desc.end(); ++iter) {
      delete iter->second;
    }

    for (map_prim_t::iterator iter = g_prim.begin(); iter != g_prim.end(); ++iter) {
      delete iter->second;
    }
}

#endif
