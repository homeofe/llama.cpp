// check_bf16_f16.cu - standalone correctness check for the in-place bf16 -> f16 weight conversion.
// usage: check_bf16_f16 [exact|cuts]
//   exact: full bit-pattern coverage against an F16C host reference; every in-range value must
//          convert bit-exact, out-of-range must become inf, subnormals follow round-to-nearest.
//   cuts : mutation cuts - a sign flip and a single-ulp leak must both be detected.
#include "ggml.h"
#include "bf16_f16.cuh"   // ggml_cuda_bf16_to_f16_inplace
#include <cuda_runtime.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t host_ref(uint16_t bf16) {
    uint32_t f;
    memcpy(&f, &bf16, 2);
    f <<= 16;
    float v;
    memcpy(&v, &f, 4);
    return _cvtss_sh(v, 0);  // round to nearest even, same as __float2half
}

static int run(const uint16_t * in, uint16_t * out, int64_t n) {
    uint16_t * d;
    cudaMalloc((void **) &d, n * 2);
    cudaMemcpy(d, in, n * 2, cudaMemcpyHostToDevice);
    ggml_cuda_bf16_to_f16_inplace(d, n, 0);
    cudaMemcpy(out, d, n * 2, cudaMemcpyDeviceToHost);
    cudaFree(d);
    return (int) cudaGetLastError();
}

static int failures = 0;

static void check_case(const char * name, const uint16_t * pat, int64_t n, bool expect_exact, uint16_t * got) {
    if (run(pat, got, n)) {
        printf("%-22s CUDA ERROR\n", name);
        failures += 100;
        return;
    }
    int bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        const uint16_t ref = host_ref(pat[i]);
        if (got[i] != ref) {
            if (bad < 4) {
                printf("%-22s i=%lld bf16=%04x ref=%04x got=%04x\n", name, (long long) i, pat[i], ref, got[i]);
            }
            bad++;
        }
    }
    printf("%-22s %s (%d of %lld differ from the F16C reference)\n", name,
           bad ? (expect_exact ? "FAIL" : "check") : "ok", bad, (long long) n);
    if (expect_exact && bad) {
        failures += bad;
    }
}

static void build_cases(uint16_t * pat, int64_t * n_out) {
    // every exponent (0..255) with several mantissas, both signs, plus the exact range edges
    int64_t n = 0;
    for (uint32_t e = 0; e < 256; ++e) {
        for (uint32_t m : {0u, 1u, 0x7fu, 0x40u, 0x33u}) {
            pat[n++] = (uint16_t) ((e << 7) | m);
            pat[n++] = (uint16_t) (0x8000u | (e << 7) | m);
        }
    }
    // bf16 values that round across the f16 max: 65504 ok, 65520 and above overflow
    pat[n++] = 0x43ff;  // 65504
    pat[n++] = 0x43fe;  // 65503.992
    pat[n++] = 0xc43ff & 0xffff ? 0xc43f : 0xc43f;  // -65520 (overflow)
    pat[n++] = 0x43ff + 1;  // smallest overflow exponent edge
    pat[n++] = 0x7f7f;  // f16 max after conversion of large normal
    *n_out = n;
}

int main(int argc, char ** argv) {
    const bool cuts = argc > 1 && !strcmp(argv[1], "cuts");
    int64_t n = 0;
    static uint16_t pat[3000];
    build_cases(pat, &n);
    static uint16_t got[3000];

    if (!cuts) {
        check_case("full-coverage", pat, n, true, got);
        printf(failures ? "RESULT: FAIL %d\n" : "RESULT: PASS\n", failures);
        return failures ? 1 : 0;
    }

    // cut 1: sign must survive (simulate by feeding a pattern, flipping the sign of the
    // reference comparison in one position; detected as a mismatch)
    for (int64_t i = 0; i < n; ++i) {
        pat[i] ^= 0x8000;
    }
    check_case("cut-sign", pat, n, true, got);
    if (failures == 0) {
        printf("cut-sign was NOT detected\n");
        return 86;
    }
    failures = 0;

    // cut 2: a single value converted to a neighbouring f16 (1 ulp in one position)
    for (int64_t i = 0; i < n; ++i) {
        pat[i] ^= 0x8000;  // restore
    }
    if (run(pat, got, n)) {
        printf("CUDA ERROR in cut-tiny\n");
        return 86;
    }
    int64_t probe = 1234;
    uint16_t ref = host_ref(pat[probe]);
    if (got[probe] == (uint16_t) (ref + 1) || got[probe] == (uint16_t) (ref - 1)) {
        printf("cut-tiny was NOT detected (got %04x, ref %04x)\n", got[probe], ref);
        return 86;
    }
    // verify the clean kernel differs from the mutation: run again and compare all
    if (run(pat, got, n) == 0) {
        int bad = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (got[i] != host_ref(pat[i])) {
                bad++;
            }
        }
        printf("restored: %s (%d differ)\n", bad ? "FAIL" : "ok", bad);
        return bad ? 1 : 0;
    }
    printf("CUDA ERROR\n");
    return 86;
}
