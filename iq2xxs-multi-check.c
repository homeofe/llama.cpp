// Compare shared IQ2_XXS decode with round-1 dots, including irregular columns and tails.
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*multi_dot_t)(int, float *, size_t, const void *, const void * const *, int);
void ggml_vec_dot_iq2_xxs_q8_K(int, float *, size_t, const void *, size_t, const void *, size_t, int);
static uint64_t state = 88172645463325252ull;
static uint32_t rnd(void) { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return (uint32_t) state; }
static float rscale(void) { return ldexpf(((int) (rnd()%2001) - 1000)/1000.0f, (int) (rnd()%9) - 6); }

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "usage: %s library symbol columns [cut_tail]\n", argv[0]);
        return 2;
    }
    const int width = atoi(argv[3]);
    if (width < 2 || width > 4) { return 2; }
    void * handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!handle) { fprintf(stderr, "%s\n", dlerror()); return 2; }
    void * symbol = dlsym(handle, argv[2]);
    if (!symbol) { fprintf(stderr, "%s\n", dlerror()); return 2; }
    multi_dot_t multi;
    memcpy(&multi, &symbol, sizeof(multi));
    ggml_cpu_init();
    const int lengths[] = {1, 2, 3, 10};
    const int columns[] = {1, width, width + 1, 2*width + 1};
    const int order[] = {8, 0, 6, 2, 7, 1, 5, 3, 4};
    block_iq2_xxs x[3][11];
    block_q8_K y[9][12];
    const void * ys[9];
    for (int c = 0; c < 9; ++c) { ys[c] = &y[order[c]][c%3]; }
    long failures = 0;
    for (int mode = 0; mode < 2; ++mode) {
        for (int len = 0; len < 4; ++len) {
            const int nb = lengths[len];
            const char * count_env = getenv("CHECK_CASES");
            const int count = count_env ? atoi(count_env) : (mode == 0 ? 25000 : 50000);
            long bad = 0, comparisons = 0;
            for (int it = 0; it < count; ++it) {
                for (int r = 0; r < 3; ++r) {
                    for (int b = 0; b < nb; ++b) {
                        x[r][b].d = ggml_fp32_to_fp16(mode == 0 ? 1.0f : rscale());
                        for (size_t j = 0; j < sizeof(x[r][b].qs); ++j) { ((uint8_t *) x[r][b].qs)[j] = rnd(); }
                    }
                }
                for (int c = 0; c < 9; ++c) {
                    block_q8_K * col = &y[order[c]][c%3];
                    for (int b = 0; b < nb; ++b) {
                        col[b].d = mode == 0 ? 1.0f : rscale();
                        for (int j = 0; j < QK_K; ++j) { col[b].qs[j] = mode == 0 ? (int) (rnd()%5) - 2 : (int) (rnd()%255) - 127; }
                        for (int g = 0; g < QK_K/16; ++g) {
                            int sum = 0;
                            for (int j = 0; j < 16; ++j) { sum += col[b].qs[16*g + j]; }
                            col[b].bsums[g] = sum;
                        }
                    }
                }
                for (int r = 0; r < 3; ++r) {
                    for (int ci = 0; ci < 4; ++ci) {
                        const int nc = columns[ci];
                        float got[9*5], ref;
                        for (int j = 0; j < 9*5; ++j) { got[j] = 1234567.0f; }
                        int c = 0;
                        for (; c + width <= nc; c += width) { multi(nb*QK_K, got + 5*c, 5, x[r], ys + c, width); }
                        for (; c < nc; ++c) {
                            if (argc != 5) { ggml_vec_dot_iq2_xxs_q8_K(nb*QK_K, got + 5*c, 0, x[r], 0, ys[c], 0, 1); }
                        }
                        for (c = 0; c < nc; ++c) {
                            ggml_vec_dot_iq2_xxs_q8_K(nb*QK_K, &ref, 0, x[r], 0, ys[c], 0, 1);
                            ++comparisons;
                            if (!isfinite(got[5*c]) || memcmp(&ref, &got[5*c], sizeof(ref))) {
                                if (bad < 3) { printf("mismatch mode=%d nb=%d it=%d row=%d nc=%d col=%d got=%.9g ref=%.9g\n", mode, nb, it, r, nc, c, got[5*c], ref); }
                                ++bad;
                            }
                        }
                        for (int j = 0; j < 9*5; ++j) { if ((j%5 || j/5 >= nc) && got[j] != 1234567.0f) { ++bad; } }
                    }
                }
            }
            printf("%s nb=%d width=%d cases=%d comparisons=%ld bitwise_failures=%ld\n", mode == 0 ? "exact" : "full", nb, width, count, comparisons, bad);
            failures += bad;
        }
    }
    printf("total_failures=%ld\n", failures);
    dlclose(handle);
    return failures != 0;
}
