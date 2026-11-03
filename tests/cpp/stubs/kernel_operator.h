// Minimal scalar-only SDK substitute for exercising the actual planner on CPU.
// This does not model device pipelines or synchronization.
#pragma once
#include <cstdint>
#define __aicore__
#define __gm__
extern bool c8_test_aiv;
extern uint32_t c8_test_block;
#define ASCEND_IS_AIV (c8_test_aiv)
namespace AscendC {
inline uint32_t GetBlockIdx() { return c8_test_block; }
template<class T> struct GlobalTensor {
    T *data;
    void SetGlobalBuffer(T *p) { data = p; }
    T GetValue(uint32_t i) const { return data[i]; }
};
}
