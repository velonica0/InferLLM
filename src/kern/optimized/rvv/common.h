#pragma once

#include <assert.h>
#include <cstdlib>
#include "core/tensor.h"
#include "file.h"
#include "kern/kernel_define.h"
#include <riscv_vector.h>  // 包含 RISC-V 矢量扩展的 C 语言接口


namespace inferllm {
namespace opt {

typedef enum SEW {
    E8 = 0,
    E16 = 1,
    E32 = 2,
    E64 = 3,
} SEW;

extern size_t vlmax[4][4];
extern size_t vlmul[4][4];

inline size_t mk_lmul(SEW sew, int sz) {
    size_t rvlmul;
    for (int i = 0; i < 4; i++) {
        rvlmul = vlmul[sew][i];
        if (sz <= vlmax[sew][i])
            break;
    }
    return rvlmul;
}

inline size_t mk_vtype(SEW sew, size_t LMUL) {
#if INFER_RVV > 107
    return (sew << 3) | LMUL;
#else
    return (sew << 2) | LMUL;
#endif
}

#if INFER_RVV > 107
#define VSET1(e, m) asm volatile("vsetivli x0, 1, " #e ", " #m);
#else
#define VSET1(e, m) asm volatile("vsetvli x0, %[sz], " #e ", " #m ::[sz] "r"(1));
#endif

inline float vmaxabs(size_t sz, const float* t, float init) {
    //VSET1(e32, m1);
    size_t vl_ = __riscv_vsetvl_e32m1(1);
    //asm volatile("vfmv.s.f v1, %[init]\n" : : [init] "f"(init));
    vfloat32m1_t v1 =  __riscv_vfmv_s_f_f32m1(init, vl_);
    size_t lmul = mk_lmul(E32, sz);
    size_t vt8 = mk_vtype(E8, lmul), vt32 = mk_vtype(E32, lmul);
    for (; sz > 0;) {
        // int vl;
        // asm volatile(
        //         "vsetvl        x0, %[sz8], %[vt8]\n"
        //         "vlbu.v        v8, (%[t])\n"
        //         "vsetvl        %[vl], %[sz32], %[vt32]\n"
        //         "vfsgnjx.vv    v8, v8, v8\n"
        //         "vfredmax.vs   v1, v8, v1\n"
        //         : [vl] "=r"(vl)
        //         : [sz8] "r"(sz * 4), [vt8] "r"(vt8), [sz32] "r"(sz), [vt32] "r"(vt32),
        //           [t] "r"(t)
        //         : "memory");
        size_t vl = __riscv_vsetvl_e32m1(sz);
        vfloat32m1_t v8 = __riscv_vle32_v_f32m1(t, vl);
        v8 = __riscv_vfsgnjx_vv_f32m1(v8, v8, vl);
        v1 = __riscv_vfredmax_vs_f32m1_f32m1(v8, v1,vl);
        t += vl;
        sz -= vl;
    }
    // VSET1(e32, m1);
    // asm volatile("vfmv.f.s  %[init],  v1\n" : [init] "=f"(init));
    init = __riscv_vfmv_f_s_f32m1_f32(v1);
    return init;
}

inline float vmax(int sz, const float* t, float init) {
    // VSET1(e32, m1);
    // asm volatile("vfmv.s.f v1, %[init]\n" : : [init] "f"(init));
    size_t vl_ = __riscv_vsetvl_e32m1(1);
    vfloat32m1_t v1 =  __riscv_vfmv_s_f_f32m1(init, vl_);
    size_t lmul = mk_lmul(E32, sz);
    size_t vt8 = mk_vtype(E8, lmul), vt32 = mk_vtype(E32, lmul);
    for (; sz > 0;) {
        // int vl;
        // asm volatile(
        //         //"vsetvl        x0, %[sz8], %[vt8]\n"
        //         //"vlbu.v        v8, (%[t])\n"
        //         //"vsetvl        %[vl], %[sz32], %[vt32]\n"
        //          "vsetvl        %[vl], %[sz], %[vt32]\n"
        //          "vle.v        v8, (%[t])\n"  
        //         "vfredmax.vs   v1, v8, v1\n"
        //         : [vl] "=r"(vl)
        //         : [sz8] "r"(sz * 4), [vt8] "r"(vt8), [sz32] "r"(sz), [vt32] "r"(vt32),
        //           [t] "r"(t)
        //         : "memory");
                // "vsetvl        %[vl], %[sz], %[vt32]\n"
                // "vle.v        v8, (%[t])\n"      
        size_t vl = __riscv_vsetvl_e32m1(sz);
        vfloat32m1_t v8 = __riscv_vle32_v_f32m1(t, vl);
        v1 = __riscv_vfredmax_vs_f32m1_f32m1(v8, v1,vl);

        t += vl;
        sz -= vl;
    }
    // VSET1(e32, m1);
    // asm volatile("vfmv.f.s  %[init],  v1\n" : [init] "=f"(init));
    init = __riscv_vfmv_f_s_f32m1_f32(v1);
    return init;
}

inline float vmulsum(const float* x, const float* y, int sz, float init) {
    // asm volatile(
    //         //"vsetvli x0, %[sz], e32, m1\n"
    //         "vsetvli x0, %[sz], e32\n"
    //         "vfmv.s.f v1, %[init]\n"
    //         :
    //         : [init] "f"(init), [sz] "r"(sz));
    // 设置矢量寄存器配置
    size_t vl = __riscv_vsetvl_e32m1(sz);
    vfloat32m1_t v1 = __riscv_vfmv_v_f_f32m1(init, vl);
    // for (; sz > 0;) {
    //     int vl = 0;
    //     asm volatile(
    //             // "slli          t0, %[sz], 2\n"
    //             // "vsetvli       t0, t0, e8, m8\n"
    //             // "vlbu.v        v8,  (%[x])\n"
    //             // "vlbu.v        v16, (%[y])\n"
    //             // "srli          t0, t0, 2\n"
    //             // "vsetvli       %[vl], t0, e32, m8\n"
    //             "vsetvli       %[vl], %[sz], e32\n"
    //             "vle.v        v8,  (%[x])\n"
    //             "vle.v        v16, (%[y])\n"
    //             "vfmul.vv      v8, v8, v16\n"
    //             "vfredsum.vs   v1, v8, v1\n"
    //             : [vl] "=r"(vl)
    //             : [sz] "r"(sz), [x] "r"(x), [y] "r"(y)
    //             : "t0", "memory");
    //     x += vl;
    //     y += vl;
    //     sz -= vl;
    // }
    while (sz > 0) {
        size_t vl = __riscv_vsetvl_e32m1(sz);
        vfloat32m1_t v8 = __riscv_vle32_v_f32m1(x, vl); // 加载 x 数据
        vfloat32m1_t v16 = __riscv_vle32_v_f32m1(y, vl); // 加载 y 数据
        v8 = __riscv_vfmul_vv_f32m1(v8, v16, vl); // 计算乘积
        v1 = __riscv_vfredusum_vs_f32m1_f32m1(v8, v1, vl); // 累加和

        x += vl; // 更新 x 指针
        y += vl; // 更新 y 指针
        sz -= vl; // 减少处理的元素数量
    }
    // asm volatile(
    //         "vsetvli x0, x0, e32, m1\n"
    //         "vfmv.f.s  %[init],  v1\n"
    //         : [init] "=f"(init));
    init = __riscv_vfmv_f_s_f32m1_f32(v1);
    return init;
}

inline float vsqrsum(const float* t, int sz, float init) {
    // asm volatile(
    //         // "vsetvli x0, %[sz], e32, m1\n"
    //         "vsetvli x0, %[sz], e32\n"
    //         "vfmv.s.f v1, %[init]\n"
    //         :
    //         : [init] "f"(init), [sz] "r"(sz));
    // 设置矢量寄存器的配置
    size_t vl = __riscv_vsetvl_e32m1(sz);
    vfloat32m1_t v1 = __riscv_vfmv_v_f_f32m1(init, vl);
    // for (; sz > 0;) {
    //     int vl = 0;
    //     asm volatile(
    //             // "slli          t0, %[sz], 2\n"
    //             // "vsetvli       t0, t0, e8, m8\n"
    //             // "vlbu.v        v8, (%[t])\n"
    //             // "srli          t0, t0, 2\n"
    //             // "vsetvli       %[vl], t0, e32, m8\n"
    //             "vsetvli       %[vl], %[sz], e32\n"
    //             "vle.v        v8, (%[t])\n"   
    //             "vfmul.vv      v8, v8, v8\n"
    //             "vfredsum.vs   v1, v8, v1\n"
    //             : [vl] "=r"(vl)
    //             : [sz] "r"(sz), [t] "r"(t)
    //             : "t0", "memory");
    //     t += vl;
    //     sz -= vl;
    // }
    while (sz > 0) {
        size_t vl = __riscv_vsetvl_e32m1(sz);
        vfloat32m1_t v8 = __riscv_vle32_v_f32m1(t, vl); // 加载数据
        v8 = __riscv_vfmul_vv_f32m1(v8, v8, vl); // 计算平方
        v1 = __riscv_vfredusum_vs_f32m1_f32m1(v8, v1, vl); // 累加和

        t += vl; // 更新数据指针
        sz -= vl; // 减少处理的元素数量
    }
    // asm volatile(
    //         "vsetvli x0, x0, e32, m1\n"
    //         "vfmv.f.s  %[init],  v1\n"
    //         : [init] "=f"(init));
    init = __riscv_vfmv_f_s_f32m1_f32(v1);
    return init;
}

inline void vscal(int sz, const float* x, float* z, float scale) {
    // for (; sz > 0;) {
    //     int vl = 0;
    //     asm volatile(
    //             // "slli          t0, %[sz], 2\n"
    //             // "vsetvli       t0, t0, e8, m8\n"
    //             // "vlbu.v        v8, (%[x])\n"
    //             // "srli          t1, t0, 2\n"
    //             // "vsetvli       %[vl], t1, e32, m8\n"
    //             // "vfmul.vf      v8, v8, %[scale]\n"
    //             // "vsetvli       x0, t0, e8, m8\n"
    //             // "vsb.v         v8, (%[z])\n"
    //             // : [vl] "=r"(vl)
    //             // : [sz] "r"(sz), [x] "r"(x), [z] "r"(z), [scale] "f"(scale)
    //             // : "t0", "t1", "memory");
    //             "vsetvli %[vl], %[sz], e32\n" 
    //             "vle.v v8, (%[x])\n" 
    //             "vfmul.vf v8, v8, %[scale]\n"
    //             "vse.v v8, (%[z])\n"
    //             : [vl] "=r" (vl)
    //             : [sz] "r" (sz), [x] "r" (x), [scale] "f" (scale), [z] "r" (z)
    //             : "memory"
    //         );
    //     x += vl;
    //     z += vl;
    //     sz -= vl;
    // }
    while (sz > 0) {
        size_t vl = __riscv_vsetvl_e32m1(sz);  // 设置矢量寄存器长度

        // 从内存加载数据到矢量寄存器
        vfloat32m1_t v8 = __riscv_vle32_v_f32m1(x, vl);

        // 将矢量寄存器中的数据乘以标量
        v8 = __riscv_vfmul_vf_f32m1(v8, scale, vl);

        // 将计算结果存储回内存
        __riscv_vse32_v_f32m1(z, v8, vl);

        // 更新指针和剩余元素数量
        x += vl;
        z += vl;
        sz -= vl;
    }
}

inline void vadd(int sz, const float* x, const float* y, float* z) {
    while(sz>0) {
        // int vl = 0;
        // asm volatile(
        //         // "slli          t0, %[sz], 2\n"
        //         // "vsetvli       t0, t0, e8, m8\n"
        //         // "vlbu.v        v8, (%[x])\n"
        //         // "vlbu.v        v16, (%[y])\n"
        //         // "srli          t1, t0, 2\n"
        //         // "vsetvli       %[vl], t1, e32, m8\n"
        //         // "vfadd.vv      v8, v8, v16\n"
        //         // "vsetvli       x0, t0, e8, m8\n"
        //         // "vsb.v         v8, (%[z])\n"
        //         // : [vl] "=r"(vl)
        //         // : [sz] "r"(sz), [x] "r"(x), [y] "r"(y), [z] "r"(z)
        //         // : "t0", "t1", "memory");
        //         "vsetvli %[vl], %[sz], e32\n"
        //         "vle.v v8, (%[x])\n"
        //         "vle.v v16, (%[y])\n"
        //         "vfadd.vv v8, v8, v16\n"
        //         "vse.v v8, (%[z])\n"
        //         : [vl] "=r" (vl)
        //         : [sz] "r" (sz), [x] "r" (x), [y] "r" (y), [z] "r" (z)
        //         : "memory"
        //     );
        size_t vl = __riscv_vsetvl_e32m1(sz);  // 设置矢量寄存器长度

        // 从内存加载数据到矢量寄存器
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(x, vl);
        vfloat32m1_t vy = __riscv_vle32_v_f32m1(y, vl);

        // 将两个矢量寄存器中的数据相加
        vfloat32m1_t vz = __riscv_vfadd_vv_f32m1(vx, vy, vl);

        // 将计算结果存储回内存
        __riscv_vse32_v_f32m1(z, vz, vl);


        x += vl;
        y += vl;
        z += vl;
        sz -= vl;
    }
}

inline void vmul(int sz, const float* x, const float* y, float* z) {
    for (; sz > 0;) {
        // int vl = 0;
        // asm volatile(
        //         // "slli          t0, %[sz], 2\n"
        //         // "vsetvli       t0, t0, e8, m8\n"
        //         // "vlbu.v        v8, (%[x])\n"
        //         // "vlbu.v        v16, (%[y])\n"
        //         // "srli          t1, t0, 2\n"
        //         // "vsetvli       %[vl], t1, e32, m8\n"
        //         // "vfmul.vv      v8, v8, v16\n"
        //         // "vsetvli       x0, t0, e8, m8\n"
        //         // "vsb.v         v8, (%[z])\n"
        //         // : [vl] "=r"(vl)
        //         // : [sz] "r"(sz), [x] "r"(x), [y] "r"(y), [z] "r"(z)
        //         // : "t0", "t1", "memory");
        //         "vsetvli %[vl], %[sz], e32\n"
        //         "vle.v v8, (%[x])\n"   
        //         "vle.v v16, (%[y])\n" 
        //         "vfmul.vv v8, v8, v16\n"   
        //         "vse.v v8, (%[z])\n"
        //         : [vl] "=r" (vl)
        //         : [sz] "r" (sz), [x] "r" (x), [y] "r" (y), [z] "r" (z) 
        //         : "memory"
        //     );
        size_t vl = __riscv_vsetvl_e32m1(sz);  // 设置矢量寄存器长度

        // 从内存加载数据到矢量寄存器
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(x, vl);
        vfloat32m1_t vy = __riscv_vle32_v_f32m1(y, vl);

        // 将两个矢量寄存器中的数据逐个相乘
        vfloat32m1_t vz = __riscv_vfmul_vv_f32m1(vx, vy, vl);

        // 将计算结果存储回内存
        __riscv_vse32_v_f32m1(z, vz, vl);

        x += vl;
        y += vl;
        z += vl;
        sz -= vl;
    }
}

void dumpV();

}  // namespace opt
}  // namespace inferllm
