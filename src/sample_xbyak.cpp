#define XBYAK_NO_OP_NAMES
#include <xbyak/xbyak.h>

//#include <bfloat16.hpp>

struct Code : Xbyak::CodeGenerator {
    const Xbyak::Reg64& src;
    const Xbyak::Reg64& dst;
    const Xbyak::Reg32& loop;
    Code()
        : src(rsi)
        , dst(rdi)
        , loop(edx)
    {
        Xbyak::Label l0;
        L(l0);

        //vmovups(zmm0, zword[src]);
        //vmovups(zword[dst], zmm0);
        //add(src, 64);
        //add(dst, 64);

        vcvtneps2bf16(ymm0, zword[src]);
        vmovups(yword[dst], ymm0);
        add(src, 64);
        add(dst, 32);

        dec(loop);
        jg(l0, T_NEAR);

        mov(eax, loop);
        ret();
    }
};


int main() {
  const int len = 32;
  float src[len];
  //float dst[len/2] = {0};
  uint16_t dst[len] = {0};
  for (int i = 0; i < len; ++i) {
    src[i] = 4.111111;
    //src[i] = 1.0;
  }

  Code c;
  int (*f)(void*, void*, int) = c.getCode<int (*)(void*, void*, int)>();
  int ret = f(dst, src, 2);

  for (int i = 0; i < len; ++i) {
    printf("%08x\n", (int)src[i]);
  }
  printf("\n\n");

  for (int i = 0; i < len; ++i) {
    printf("%d\n", ((short *)dst)[i]);
  }

  printf("src=%p, dst = %p, dst[0]=%f, ret = %d\n", src, dst, dst[0], ret);

  for (int i = 0; i < len; ++i) {
    printf("%x\n", *(uint32_t *)&src[0]);
  }
  for (int i = 0; i < len; ++i) {
    printf("%x\n", *(uint16_t *)&dst[0]);
  }

/*
  mkldnn::impl::bfloat16_t out[len];
  mkldnn::impl::cvt_float_to_bfloat16(out, src, len);
  for (int i = 0; i < len; ++i) {
    printf("%x\n", *(uint16_t *)&out[0]);
  }
*/

}

