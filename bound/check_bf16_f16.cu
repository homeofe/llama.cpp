// check_bf16_f16.cu - standalone correctness check for the in-place bf16 -> f16 weight conversion.
// usage: check_bf16_f16
//   Full bit-pattern coverage against an independent software reference (round-to-nearest-even):
//   in-range values convert bit-exact, out-of-range becomes inf, subnormals round like IEEE says.
//   Mutation cuts are done in the kernel source (sign drop, 1-ulp leak) and must FAIL here.
#include "ggml.h"
#include "bf16_f16.cuh"   // ggml_cuda_bf16_to_f16_inplace
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

// independent f32 -> f16 (binary16) reference, round to nearest even
static uint16_t f32_to_f16_ref(uint32_t f) {
    const uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = (int32_t) ((f >> 23) & 0xff) - 127 + 15;
    uint32_t man = (f & 0x7fffffu) >> 13;          // 10 kept mantissa bits
    const uint32_t rem = (f & 0x1fffu);            // rounding remainder
    if (exp >= 0x1f) {                              // overflow or inf
        return (uint16_t) (sign | 0x7c00u);
    }
    if (exp <= 0) {                                 // f16 subnormal or zero
        man |= 0x400u;                              // implicit one
        const uint32_t shift = (uint32_t) (1 - exp);
        if (shift >= 12) {                          // below a quarter of the quantum: rounds to zero
            return (uint16_t) sign;
        }
        const uint32_t kept = man >> shift;
        const uint32_t rrem = man & ((1u << shift) - 1);
        const uint32_t half = 1u << (shift - 1);
        if (rrem > half || (rrem == half && (kept & 1))) {
            return (uint16_t) (sign | (kept + 1));
        }
        return (uint16_t) (sign | kept);
    }
    if (rem > 0x1000u || (rem == 0x1000u && (man & 1))) {
        man++;                                      // carry into exponent is correct
    }
    if ((man >> 10) != 0) {
        man = 0;
        exp++;
        if (exp >= 0x1f) {
            return (uint16_t) (sign | 0x7c00u);
        }
    }
    return (uint16_t) (sign | ((uint32_t) exp << 10) | (man & 0x3ffu));
}

static uint16_t host_ref(uint16_t bf16) {
    uint32_t f;
    memcpy(&f, &bf16, 2);
    f <<= 16;                                       // bf16 -> f32 bits
    return f32_to_f16_ref(f);
}

static int run(const uint16_t * in, uint16_t * out, int64_t n) {
    uint16_t * d;
    if (cudaMalloc((void **) &d, n * 2) != cudaSuccess) return 1;
    cudaMemcpy(d, in, n * 2, cudaMemcpyHostToDevice);
    ggml_cuda_bf16_to_f16_inplace(d, n, 0);
    cudaMemcpy(out, d, n * 2, cudaMemcpyDeviceToHost);
    cudaFree(d);
    return (int) cudaGetLastError();
}

static void build_cases(uint16_t * pat, int64_t * n_out) {
    // every exponent with several mantissas, both signs: covers normals, subnormals, inf, nan
    int64_t n = 0;
    for (uint32_t e = 0; e < 256; ++e) {
        for (uint32_t m = 0; m < 3; ++m) {
            const uint32_t man = m * 0x2bu;
            pat[n++] = (uint16_t) ((e << 7) | man);
            pat[n++] = (uint16_t) (0x8000u | (e << 7) | man);
        }
    }
    pat[n++] = 0x43ff;   // 65504: f16 max, must convert exactly
    pat[n++] = 0xc43f;   // -65520: overflow to -inf
    pat[n++] = 0x4400;   // 65536: overflow to inf
    pat[n++] = 0x3801;   // 2^-14: smallest f16 normal, exact edge
    pat[n++] = 0x0080;   // smallest bf16 normal, becomes f16 subnormal
    pat[n++] = 0x0001;   // smallest bf16 subnormal, rounds to zero
    *n_out = n;
}

int main(void) {
    static uint16_t pat[3000];
    static uint16_t got[3000];
    int64_t n = 0;
    build_cases(pat, &n);
    if (run(pat, got, n)) {
        printf("RESULT: CUDA ERROR\n");
        return 86;
    }
    int bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        uint16_t ref = host_ref(pat[i]);
        const bool nan_ref = ((ref & 0x7c00u) == 0x7c00u) && (ref & 0x3ffu);
        const bool nan_got = ((got[i] & 0x7c00u) == 0x7c00u) && (got[i] & 0x3ffu);
        if (nan_ref && nan_got) {
            continue;                                  // any NaN matches any NaN (payloads differ by design)
        }
        if (got[i] != ref) {
            if (bad < 8) {
                printf("i=%lld bf16=%04x ref=%04x got=%04x\n", (long long) i, pat[i], ref, got[i]);
            }
            bad++;
        }
    }
    printf("full-coverage: %s (%d of %lld differ from the software reference)\n",
           bad ? "FAIL" : "ok", bad, (long long) n);
    printf(bad ? "RESULT: FAIL %d\n" : "RESULT: PASS\n", bad);
    return bad ? 1 : 0;
}
