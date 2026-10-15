// Compare IQ2_XXS AVX2 against the generic kernel, following ref/iq3_check.c.
// Exact mode uses unit scales and [-2, 2]; full range uses finite scales and [-127, 127].
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*vec_dot_t)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_iq2_xxs_q8_K(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_iq2_xxs_q8_K_generic(int, float *, size_t, const void *, size_t, const void *, size_t, int);

static uint64_t rs = 88172645463325252ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t) rs; }
static float rscale(void) { return ldexpf(((int) (rnd() % 2001) - 1000) / 1000.0f, (int) (rnd() % 9) - 6); }

int main(int argc, char ** argv) {
    vec_dot_t fast = ggml_vec_dot_iq2_xxs_q8_K;
    void * handle = NULL;
    if (argc == 3) {
        handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!handle) { fprintf(stderr, "%s\n", dlerror()); return 2; }
        void * symbol = dlsym(handle, argv[2]);
        if (!symbol) { fprintf(stderr, "%s\n", dlerror()); return 2; }
        memcpy(&fast, &symbol, sizeof(fast));
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [variant.so symbol]\n", argv[0]);
        return 2;
    }
    struct ggml_context * ctx = ggml_init((struct ggml_init_params) {1024*1024, NULL, true});
    if (!ctx) { return 2; }
    ggml_cpu_init();
    const int nbs[] = {1, 2, 3, 10};
    block_iq2_xxs x[10];
    block_q8_K y[10];
    unsigned char grids[256] = {0}, signs[128] = {0}, scales[16] = {0};
    long failures = 0;
    memset(x, 0, sizeof(x));
    memset(y, 0, sizeof(y));
    for (int b = 0; b < 10; ++b) {
        x[b].d = ggml_fp32_to_fp16(1.0f);
        y[b].d = 1.0f;
        memset(y[b].qs, 1, sizeof(y[b].qs));
        for (int k = 0; k < QK_K/16; ++k) y[b].bsums[k] = 16;
    }
    for (int len = 0; len < 4; ++len) {
        float a = 0, g = 0;
        const int n = nbs[len]*QK_K;
        fast(n, &a, 0, x, 0, y, 0, 1);
        ggml_vec_dot_iq2_xxs_q8_K_generic(n, &g, 0, x, 0, y, 0, 1);
        printf("known answer nb=%d: avx2 %g generic %g expected %d\n", nbs[len], a, g, n);
        failures += a != n || g != n;
    }
    for (int mode = 0; mode < 2; ++mode) {
        for (int len = 0; len < 4; ++len) {
            const int nb = nbs[len];
            const int count = mode == 0 ? 25000 : 50000;
            long bad = 0;
            double worst = 0;
            for (int it = 0; it < count; ++it) {
                double mag = 0;
                for (int b = 0; b < nb; ++b) {
                    for (size_t k = 0; k < sizeof(x[b].qs); ++k) ((uint8_t *) x[b].qs)[k] = (uint8_t) rnd();
                    x[b].d = ggml_fp32_to_fp16(mode == 0 ? 1.0f : rscale());
                    y[b].d = mode == 0 ? 1.0f : rscale();
                    for (int k = 0; k < QK_K; ++k) y[b].qs[k] = (int8_t) (mode == 0 ? (int) (rnd() % 5) - 2 : (int) (rnd() % 255) - 127);
                    for (int g = 0; g < QK_K/16; ++g) {
                        int sum = 0;
                        for (int k = 0; k < 16; ++k) sum += y[b].qs[16*g+k];
                        y[b].bsums[g] = (int16_t) sum;
                    }
                    const double d = fabs((double) ggml_fp16_to_fp32(x[b].d) * y[b].d) * 0.125;
                    for (int g = 0; g < QK_K/32; ++g) {
                        uint32_t words[2];
                        memcpy(words, x[b].qs + 4*g, sizeof(words));
                        const uint8_t * idx = (const uint8_t *) words;
                        const int scale = words[1] >> 28;
                        scales[scale] = 1;
                        for (int l = 0; l < 4; ++l) {
                            grids[idx[l]] = 1;
                            signs[(words[1] >> (7*l)) & 127] = 1;
                            const uint8_t * grid = (const uint8_t *) &iq2xxs_grid[idx[l]];
                            for (int j = 0; j < 8; ++j) mag += d * (2*scale+1) * grid[j] * abs(y[b].qs[32*g+8*l+j]);
                        }
                    }
                }
                float a = 0, g = 0;
                fast(nb*QK_K, &a, 0, x, 0, y, 0, 1);
                ggml_vec_dot_iq2_xxs_q8_K_generic(nb*QK_K, &g, 0, x, 0, y, 0, 1);
                const double err = mag > 0 ? fabs((double) a-g)/mag : (a == g ? 0 : INFINITY);
                if (err > worst) worst = err;
                if (!isfinite(a) || !isfinite(g) || (mode == 0 ? a != g : err > 1e-6)) {
                    if (bad < 3) printf("mismatch mode=%d nb=%d case=%d: avx2 %.9g generic %.9g normalized %.9g\n", mode, nb, it, a, g, err);
                    ++bad;
                }
            }
            printf("%s nb=%d: %d cases, %ld failures, worst abs_error/sum_abs_terms %.9g\n", mode == 0 ? "exact" : "full", nb, count, bad, worst);
            failures += bad;
        }
    }
    int ng = 0, ns = 0, nl = 0;
    for (int i = 0; i < 256; ++i) ng += grids[i];
    for (int i = 0; i < 128; ++i) ns += signs[i];
    for (int i = 0; i < 16; ++i) nl += scales[i];
    printf("coverage: grids %d/256, signs %d/128, scales %d/16; failures %ld\n", ng, ns, nl, failures);
    ggml_free(ctx);
    if (handle) dlclose(handle);
    return failures || ng != 256 || ns != 128 || nl != 16;
}
