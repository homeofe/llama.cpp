// mma8_bench.cu - standalone int8-MMA prototype for the Q6_K small-batch matvec path
// (round 13). One warp owns 16 rows x the full K; per 256-value super-block it dequantizes
// the weights to int8 in shared memory (natural k order, verified against the ggml CPU
// dequantizer), quantized activations sit in shared as q8_1, and the dot products run on
// the INT8 tensor cores via mma.m8n8k16 (Turing: 2x per mma() call, see ggml mma.cuh).
// Scale handling follows the MMQ trick: one mma covers 16 k values = exactly one Q6_K
// scale group, so the int8 scale is applied to the int32 C fragment after the mma and the
// super-block d and the q8_1 d8 are folded around it. usage:
//   mma8_bench <m> <k> <n_tok> <iters> <mode>   mode: bench | check
// check compares against a double-precision reference with the same integer math
// (raw int8 weight values x quantized activations), scale group by scale group.
// Mutation cuts (must all FAIL the check): -DMUT_OFFBY1 -DMUT_SIGNHI -DMUT_NO_SCALE -DMUT_QH_IDX
#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mma.cuh"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace ggml_cuda_mma;

// exported from libggml-cpu with C linkage
extern "C" void quantize_row_q6_K(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);

#ifdef MUT_QH_IDX
#define MMA8_MUTQ 1
#else
#define MMA8_MUTQ 0
#endif

// little-endian int32 at 4*i from a 2-byte aligned base (block_q6_K fields sit at 2 mod 4)
static __device__ __forceinline__ int mma8_get_int2(const unsigned char * p, const int i) {
    const uint16_t * p16 = (const uint16_t *) (p + 4*i);
    return (int) p16[0] | ((int) p16[1] << 16);
}

__global__ void __launch_bounds__(32, 1) mma8_q6k_matvec(
        const unsigned char * __restrict__ w, const int wblk, // super-blocks of q6_K per row
        const unsigned char * __restrict__ y,                 // q8_1 blocks, n_tok rows of k/32
        float * __restrict__ dst, const int m, const int n_tok, const int nsb, const int bmode) {

    // fragment-perfect lane assignment: the m8n8k16 A fragment puts row l/4, k-bytes
    // 4*(l%4)..+3 into lane l, so lane l unpacks rows l/4 and l/4+8 at ql int32s
    // c+4j and holds exactly the registers its own mma steps consume: no shared staging.
    const int lane = threadIdx.x;
    const int i0   = blockIdx.x * 16;
    const int c    = lane & 3;   // slot phase, all slots this lane produces are == c mod 4
    const int rr0  = lane >> 2;  // first row of this lane, second is rr0+8
    // bmode: 0 = full, 1 = dequant only (skips the mma phase), 2 = mma only (one dequant)

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    __shared__ float yd_s[8*8]; // q8_1 d per token per 32-value chunk of the super-block

    const block_q6_K * __restrict__ wbase = (const block_q6_K *) w;
    const int yb = nsb * 8; // q8_1 blocks per token

    // prefetch registers for one super-block, all loads independent
    int qlr[8], qhr[4], qlr2[8], qhr2[4]; // raw int32s of rows rr0 and rr0+8
    int yr[16];                            // token rr0, 16 int32s at slots == c mod 4
    int ydh[4];                            // token rr0, the 8 chunk d halves packed 2 per int
    int scr0[4], scr1[4];                  // scale int32s of rows rr0 and rr0+8
    int dh0, dh1;                          // the d halves of rows rr0 and rr0+8

    int o_lo[8], o_hi[8], o_lo2[8], o_hi2[8]; // unpacked int8 pairs of both rows

    const int row0 = min(i0 + rr0,     m - 1);
    const int row1 = min(i0 + rr0 + 8, m - 1);

    // prologue: prefetch super-block 0
    {
        const block_q6_K * bx0 = wbase + (int64_t) row0*wblk;
        const block_q6_K * bx1 = wbase + (int64_t) row1*wblk;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            qlr[j]  = mma8_get_int2(bx0->ql, c + 4*j);
            qlr2[j] = mma8_get_int2(bx1->ql, c + 4*j);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            qhr[j]  = mma8_get_int2(bx0->qh, c + 4*j - MMA8_MUTQ);
            qhr2[j] = mma8_get_int2(bx1->qh, c + 4*j - MMA8_MUTQ);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            scr0[j] = mma8_get_int2((const unsigned char *) bx0->scales, j);
            scr1[j] = mma8_get_int2((const unsigned char *) bx1->scales, j);
        }
        dh0 = (int) *(const unsigned short *) (&bx0->d);
        dh1 = (int) *(const unsigned short *) (&bx1->d);
        if (rr0 < n_tok) {
            #pragma unroll
            for (int cc = 0; cc < 4; ++cc) {
                const block_q8_1 * b0 = (const block_q8_1 *) y + (int64_t) rr0*yb + 2*cc;
                const block_q8_1 * b1 = b0 + 1;
                yr[4*cc + 0] = *(const int *) (b0->qs + 4*c);
                yr[4*cc + 1] = *(const int *) (b0->qs + 4*(c + 4));
                yr[4*cc + 2] = *(const int *) (b1->qs + 4*c);
                yr[4*cc + 3] = *(const int *) (b1->qs + 4*(c + 4));
                ydh[cc] = (int) *(const unsigned short *) (&b0->d) | ((int) *(const unsigned short *) (&b1->d) << 16);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 16; ++j) yr[j] = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j)  ydh[j] = 0;
        }
    }

    for (int ksb = 0; ksb < nsb; ++ksb) {
        // 1) unpack the prefetched super-block into the fragment registers
        if (!(bmode == 2 && ksb > 0)) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int slo = ((j >> 1) & 1) ? 2 : 0; // shift pair by ql int32 group
                const int shi = slo + 4;
                const int qhi = (j & 1) + ((j >> 2) << 1); // qh int32 per ql int32 group
                const int qh  = qhr[qhi];
                const int qh2 = qhr2[qhi];
                {
                    const int ql = qlr[j];
                    const int ql0 = ql & 0x0F0F0F0F;
                    const int ql1 = (ql >> 4) & 0x0F0F0F0F;
                    const int q0  = ql0 | (((qh >> slo) & 0x03030303) << 4);
                    const int q1  = ql1 | (((qh >> shi) & 0x03030303) << 4);
#ifdef MUT_OFFBY1
                    o_lo[j] = __vsub4(q0, 0x1F1F1F1F);
                    o_hi[j] = __vsub4(q1, 0x1F1F1F1F);
#elif defined(MUT_SIGNHI)
                    o_lo[j] = __vsub4(q0, 0x20202020);
                    o_hi[j] = __vsub4(0x20202020, q1);
#else
                    o_lo[j] = __vsub4(q0, 0x20202020);
                    o_hi[j] = __vsub4(q1, 0x20202020);
#endif
                }
                {
                    const int ql = qlr2[j];
                    const int ql0 = ql & 0x0F0F0F0F;
                    const int ql1 = (ql >> 4) & 0x0F0F0F0F;
                    const int q0  = ql0 | (((qh2 >> slo) & 0x03030303) << 4);
                    const int q1  = ql1 | (((qh2 >> shi) & 0x03030303) << 4);
#ifdef MUT_OFFBY1
                    o_lo2[j] = __vsub4(q0, 0x1F1F1F1F);
                    o_hi2[j] = __vsub4(q1, 0x1F1F1F1F);
#elif defined(MUT_SIGNHI)
                    o_lo2[j] = __vsub4(q0, 0x20202020);
                    o_hi2[j] = __vsub4(0x20202020, q1);
#else
                    o_lo2[j] = __vsub4(q0, 0x20202020);
                    o_hi2[j] = __vsub4(q1, 0x20202020);
#endif
                }
            }
            // d8 of this lane's token into shared; 4 lanes cover one token's 8 chunks
            if (rr0 < n_tok) {
                #pragma unroll
                for (int cc = 0; cc < 4; ++cc) {
                    yd_s[rr0*8 + 2*cc + 0] = __half2float(__ushort_as_half((unsigned short) (ydh[cc] & 0xFFFF)));
                    yd_s[rr0*8 + 2*cc + 1] = __half2float(__ushort_as_half((unsigned short) (ydh[cc] >> 16)));
                }
            } else {
                #pragma unroll
                for (int cc = 0; cc < 4; ++cc) {
                    yd_s[rr0*8 + 2*cc + 0] = 0.f;
                    yd_s[rr0*8 + 2*cc + 1] = 0.f;
                }
            }
        }
        if (bmode == 4 && ksb == 0) {
            int * di = (int *) dst;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int slo = (j < 4) ? c + 4*j : c + 4*j + 16;
                const int shi = (j < 4) ? c + 4*j + 16 : c + 4*j + 32;
                di[rr0*64 + slo]      = o_lo[j];
                di[rr0*64 + shi]      = o_hi[j];
                di[(rr0+8)*64 + slo]  = o_lo2[j];
                di[(rr0+8)*64 + shi]  = o_hi2[j];
            }
            const int lbase = 1024 + (rr0*4 + c)*12;
            const int ybase = 2048 + (rr0*4 + c)*16;
            #pragma unroll
            for (int t16 = 0; t16 < 16; ++t16) {
                di[ybase + t16] = yr[t16];
            }
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                di[2560 + (rr0*4 + c)*4 + q] = ydh[q];
            }
            di[2752 + rr0*4 + c] = dh0;
            di[2752 + 32 + rr0*4 + c] = dh1;
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                di[2752 + 64 + (rr0*4 + c)*4 + q] = scr0[q];
                di[2752 + 64 + 128 + (rr0*4 + c)*4 + q] = scr1[q];
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                di[lbase + j] = qlr[j];
                di[lbase + 8 + j] = qlr2[j];
            }
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                di[lbase + 16 + q] = qhr[q];
                di[lbase + 20 + q] = qhr2[q];
            }
            return;
        }
        __syncwarp();

        // 2) prefetch the next super-block's weights: the loads fly during the mma phase
        if (ksb + 1 < nsb && bmode != 2) {
            const block_q6_K * bx0 = wbase + (int64_t) row0*wblk + ksb + 1;
            const block_q6_K * bx1 = wbase + (int64_t) row1*wblk + ksb + 1;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                qlr[j]  = mma8_get_int2(bx0->ql, c + 4*j);
                qlr2[j] = mma8_get_int2(bx1->ql, c + 4*j);
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                qhr[j]  = mma8_get_int2(bx0->qh, c + 4*j - MMA8_MUTQ);
                qhr2[j] = mma8_get_int2(bx1->qh, c + 4*j - MMA8_MUTQ);
            }
        }

        // 3) mma over the 256 values: 16 steps of 16 k values, batch 4-5 tokens in one n=8 tile
        // bmode 1 measures the dequant phase alone
        const float d0 = __half2float(__ushort_as_half((unsigned short) dh0));
        const float d1 = __half2float(__ushort_as_half((unsigned short) dh1));
        float t32[4] = {0.f, 0.f, 0.f, 0.f};
        if (bmode != 1) {
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            tile<16, 4, int> A;
            tile< 8, 4, int> B;
            if (t < 4) {
                A.x[0] = o_lo[t];
                A.x[1] = o_lo2[t];
            } else if (t < 8) {
                A.x[0] = o_hi[t - 4];
                A.x[1] = o_hi2[t - 4];
            } else if (t < 12) {
                A.x[0] = o_lo[t - 4];
                A.x[1] = o_lo2[t - 4];
            } else {
                A.x[0] = o_hi[t - 8];
                A.x[1] = o_hi2[t - 8];
            }
            B.x[0] = yr[2*(t >> 1) + (t & 1)];
            tile<16, 8, int> C;
            mma(C, A, B);

            const int g = t;
            const int b0 = (scr0[g >> 2] >> (8*(g & 3))) & 0xFF;
            const int b1 = (scr1[g >> 2] >> (8*(g & 3))) & 0xFF;
#ifdef MUT_NO_SCALE
            const float s0 = d0;
            const float s1 = d1;
#else
            const float s0 = (float) (int8_t) b0 * d0;
            const float s1 = (float) (int8_t) b1 * d1;
#endif
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                t32[l] += (float) C.x[l] * (l < 2 ? s0 : s1);
            }
            if (t & 1) {
                const int g32 = t >> 1;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int j = 2*(lane & 3) + (l & 1);
                    acc[l] += t32[l] * yd_s[j*8 + g32];
                    t32[l]  = 0.f;
                }
            }
        }
        }

        // 4) prefetch the next super-block's activations and scales: the mma phase above
        //    has consumed the current ones, the loads land during the next unpack
        if (ksb + 1 < nsb && bmode != 2) {
            const block_q6_K * bx0 = wbase + (int64_t) row0*wblk + ksb + 1;
            const block_q6_K * bx1 = wbase + (int64_t) row1*wblk + ksb + 1;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                scr0[j] = mma8_get_int2((const unsigned char *) bx0->scales, j);
                scr1[j] = mma8_get_int2((const unsigned char *) bx1->scales, j);
            }
            dh0 = (int) *(const unsigned short *) (&bx0->d);
            dh1 = (int) *(const unsigned short *) (&bx1->d);
            if (rr0 < n_tok) {
                #pragma unroll
                for (int cc = 0; cc < 4; ++cc) {
                    const block_q8_1 * b0 = (const block_q8_1 *) y + (int64_t) rr0*yb + (ksb + 1)*8 + 2*cc;
                    const block_q8_1 * b1 = b0 + 1;
                    yr[4*cc + 0] = *(const int *) (b0->qs + 4*c);
                    yr[4*cc + 1] = *(const int *) (b0->qs + 4*(c + 4));
                    yr[4*cc + 2] = *(const int *) (b1->qs + 4*c);
                    yr[4*cc + 3] = *(const int *) (b1->qs + 4*(c + 4));
                    ydh[cc] = (int) *(const unsigned short *) (&b0->d) | ((int) *(const unsigned short *) (&b1->d) << 16);
                }
            }
        }
    }

    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int i = i0 + 8*(l >> 1) + (lane >> 2);
        const int j = 2*(lane & 3) + (l & 1);
        if (i < m && j < n_tok) {
            dst[(int64_t) i*n_tok + j] = acc[l];
        }
    }
}

// host q8_1 quantizer, identical semantics to ggml's quantize_row_q8_1_ref
static void host_quant_q8_1(const float * x, block_q8_1 * y, int k) {
    const int nb = k / 32;
    for (int i = 0; i < nb; ++i) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[i*32 + j]));
        const float d  = amax / 127.f;
        const float id = d ? 1.f/d : 0.f;
        y[i].d = ggml_fp32_to_fp16(d);
        int sum = 0;
        for (int j = 0; j < 16; ++j) {
            const int q0 = (int) roundf(x[i*32 + j]       * id);
            const int q1 = (int) roundf(x[i*32 + 16 + j]  * id);
            y[i].qs[j]      = (int8_t) q0;
            y[i].qs[16 + j] = (int8_t) q1;
            sum += q0 + q1;
        }
        y[i].s = ggml_fp32_to_fp16((float) sum * d);
    }
}

// host transcription of ggml's dequantize_row_q6_K; also emits the raw integer weight values
// (q - 32, before scales), independent of the kernel's int32-based unpack
static void host_dequant_q6_k(const block_q6_K * x_in, float * y, int8_t * q_raw, int k) {
    const int nb = k / 256;
    const block_q6_K * x = x_in;
    for (int i = 0; i < nb; ++i) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        const uint8_t * ql = x[i].ql;
        const uint8_t * qh = x[i].qh;
        const int8_t  * sc = x[i].scales;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l/16;
                const int8_t q1 = (int8_t) ((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t) ((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t) ((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[n + l +  0] = d * sc[is + 0] * q1;
                y[n + l + 32] = d * sc[is + 2] * q2;
                y[n + l + 64] = d * sc[is + 4] * q3;
                y[n + l + 96] = d * sc[is + 6] * q4;
                q_raw[n + l +  0] = q1;
                q_raw[n + l + 32] = q2;
                q_raw[n + l + 64] = q3;
                q_raw[n + l + 96] = q4;
            }
            ql += 64; qh += 32; sc += 8;
        }
        y      += 256;
        q_raw  += 256;
    }
}

static double drand(unsigned int * seed) {
    *seed = *seed * 1664525u + 1013904223u;
    return (*seed >> 8) / 16777216.0;
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <m> <k> <n_tok> <iters> <bench|check>\n", argv[0]);
        return 2;
    }
    const int m     = atoi(argv[1]);
    const int k     = atoi(argv[2]);
    const int n_tok = atoi(argv[3]);
    const int iters = atoi(argv[4]);
    const int check = !strcmp(argv[5], "check");
    const int bmode = !strcmp(argv[5], "bench-deq") ? 1 : (!strcmp(argv[5], "bench-mma") ? 2 : (!strcmp(argv[5], "dump") ? 4 : 0));

    if (k % 256 != 0 || n_tok < 1 || n_tok > 8) {
        fprintf(stderr, "k must be a multiple of 256, 1 <= n_tok <= 8\n");
        return 2;
    }
    const int nsb  = k / 256;
    const int yb   = k / 32;

    // weights: realistic magnitude, quantized with ggml's own q6_K quantizer
    float * hx      = (float *) malloc((size_t) m*k*sizeof(float));
    block_q6_K * wq = (block_q6_K *) malloc((size_t) m*nsb*sizeof(block_q6_K));
    unsigned int seed = 42 + m + k;
    for (int64_t i = 0; i < (int64_t) m*k; ++i) {
        hx[i] = (float) (0.1*drand(&seed) - 0.05);
    }
    for (int i = 0; i < m; ++i) {
        quantize_row_q6_K(hx + (int64_t) i*k, wq + (int64_t) i*nsb, k);
    }

    // activations
    float * hact = (float *) malloc((size_t) n_tok*k*sizeof(float));
    for (int64_t i = 0; i < (int64_t) n_tok*k; ++i) {
        hact[i] = (float) (drand(&seed) - 0.5);
    }
    block_q8_1 * yq = (block_q8_1 *) malloc((size_t) n_tok*yb*sizeof(block_q8_1));
    for (int t = 0; t < n_tok; ++t) {
        host_quant_q8_1(hact + (int64_t) t*k, yq + (int64_t) t*yb, k);
    }

    float * dst = (float *) malloc((size_t) m*n_tok*sizeof(float));

    unsigned char * dw = NULL; unsigned char * dy = NULL; float * ddst = NULL;
    const size_t nb_w = (size_t) m*nsb*sizeof(block_q6_K);
    const size_t nb_y = (size_t) n_tok*yb*sizeof(block_q8_1);
    cudaMalloc(&dw,   nb_w);
    cudaMalloc(&dy,   nb_y);
    cudaMalloc(&ddst, (size_t) m*n_tok*sizeof(float));
    cudaMemcpy(dw, wq, nb_w, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, yq, nb_y, cudaMemcpyHostToDevice);

    const int nblk  = (m + 15) / 16;
    const dim3 grid(nblk), block(32);
    if (getenv("MMA8_CARVEOUT")) {
        cudaFuncSetAttribute(mma8_q6k_matvec, cudaFuncAttributePreferredSharedMemoryCarveout, atoi(getenv("MMA8_CARVEOUT")));
    }

    if (bmode == 4) {
        cudaMemset(ddst, 0, (size_t) m*n_tok*sizeof(float));
        mma8_q6k_matvec<<<dim3(1), block>>>(dw, nsb, dy, ddst, m, n_tok, nsb, bmode);
        if (cudaDeviceSynchronize() != cudaSuccess) { fprintf(stderr, "kernel failed\n"); return 1; }
        cudaMemcpy(dst, ddst, (size_t) m*n_tok*sizeof(float), cudaMemcpyDeviceToHost);
        int8_t * qraw = (int8_t *) malloc((size_t) k);
        float  * ybuf = (float  *) malloc((size_t) k*sizeof(float));
        int * di = (int *) dst;
        int n_bad = 0;
        int n_load_bad = 0;
        for (int r = 0; r < 16; ++r) {
            const int rr = (r < 8) ? r : r - 8; // dumped rows use rr0 and rr0+8
            const block_q6_K * bxr = wq + (int64_t) ((r < 8) ? r : r)*nsb;
            host_dequant_q6_k(wq + (int64_t) r*nsb, ybuf, qraw, k);
            const block_q6_K * bx = wq + (int64_t) r*nsb;
            for (int c2 = 0; c2 < 4; ++c2) {
                for (int j = 0; j < 8; ++j) {
                    const int i = c2 + 4*j;
                    const int expq = (int) *(const unsigned short *) (bx->ql + 4*i) | ((int) *(const unsigned short *) (bx->ql + 4*i + 2) << 16);
                    const int lbase = 1024 + (r*4 + c2)*12;
                    if (di[lbase + j] != expq) {
                        if (n_load_bad < 2) {
                            printf("LOADBAD r=%d c=%d j=%d got=%08x exp=%08x\n", r, c2, j, di[lbase + j], expq);
                            // search the whole wq and yq buffers for the got word
                            const int target = di[lbase + j];
                            const unsigned short * w16 = (const unsigned short *) wq;
                            const int nw16 = (int) ((size_t) m*nsb*sizeof(block_q6_K) / 2);
                            int found = -1;
                            for (int t = 0; t + 1 < nw16; ++t) {
                                if (((int) w16[t] | ((int) w16[t+1] << 16)) == target) { found = t; break; }
                            }
                            printf("  got word found at wq ushort index %d (byte %d)\n", found, found*2);
                        }
                        n_load_bad++;
                    }
                }
            }
            for (int sl = 0; sl < 64; ++sl) {
                const int got = di[r*64 + sl];
                int exp = 0;
                for (int b = 0; b < 4; ++b) exp |= ((int) (unsigned char) qraw[4*sl + b]) << (8*b);
                if (got != exp) {
                    if (n_bad < 12) {
                        printf("row %d slot %2d: got %08x expect %08x (bytes got %d %d %d %d exp %d %d %d %d)\n",
                               r, sl, got, exp,
                               got & 0xFF, (got>>8)&0xFF, (got>>16)&0xFF, (got>>24)&0xFF,
                               exp & 0xFF, (exp>>8)&0xFF, (exp>>16)&0xFF, (exp>>24)&0xFF);
                    }
                    n_bad++;
                }
            }
        }
        int n_y_bad = 0;
        for (int rr = 0; rr < 8; ++rr) {
            for (int c2 = 0; c2 < 4; ++c2) {
                const int ybase = 2048 + (rr*4 + c2)*16;
                for (int cc = 0; cc < 4; ++cc) {
                    for (int h = 0; h < 2; ++h) {
                        const block_q8_1 * b = (const block_q8_1 *) yq + (int64_t) rr*yb + 2*cc + h;
                        const int e0 = *(const int *) (b->qs + 4*c2);
                        const int e1 = *(const int *) (b->qs + 4*(c2 + 4));
                        if (di[ybase + 4*cc + 2*h + 0] != e0) { if (n_y_bad < 6) printf("YBAD rr=%d c=%d cc=%d h=%d j0 got=%08x exp=%08x\n", rr, c2, cc, h, di[ybase + 4*cc + 2*h + 0], e0); n_y_bad++; }
                        if (di[ybase + 4*cc + 2*h + 1] != e1) { if (n_y_bad < 6) printf("YBAD rr=%d c=%d cc=%d h=%d j1 got=%08x exp=%08x\n", rr, c2, cc, h, di[ybase + 4*cc + 2*h + 1], e1); n_y_bad++; }
                    }
                }
            }
        }
        printf("dump: %d/1024 slots wrong, %d raw-load mismatches (rows 0..7 only), %d yr mismatches\n", n_bad, n_load_bad, n_y_bad);
        if (n_bad) {
            for (int r = 0; r < 16; ++r) {
                host_dequant_q6_k(wq + (int64_t) r*nsb, ybuf, qraw, k);
                for (int sl = 0; sl < 64; ++sl) {
                    const int got = di[r*64 + sl];
                    int exp = 0;
                    for (int b = 0; b < 4; ++b) exp |= ((int) (unsigned char) qraw[4*sl + b]) << (8*b);
                    printf("D r=%d s=%d got=%d,%d,%d,%d exp=%d,%d,%d,%d\n", r, sl,
                           (signed char) (got & 0xFF), (signed char) ((got>>8)&0xFF), (signed char) ((got>>16)&0xFF), (signed char) ((got>>24)&0xFF),
                           (signed char) (exp & 0xFF), (signed char) ((exp>>8)&0xFF), (signed char) ((exp>>16)&0xFF), (signed char) ((exp>>24)&0xFF));
                }
            }
        }
        return n_bad ? 1 : 0;
    }

    if (check) {
        cudaMemset(ddst, 0, (size_t) m*n_tok*sizeof(float));
        mma8_q6k_matvec<<<grid, block>>>(dw, nsb, dy, ddst, m, n_tok, nsb, bmode);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        cudaMemcpy(dst, ddst, (size_t) m*n_tok*sizeof(float), cudaMemcpyDeviceToHost);

        // reference: double, but the same integer math as the kernel, group by group
        float  * wrow = (float  *) malloc((size_t) k*sizeof(float));
        int8_t * qraw = (int8_t *) malloc((size_t) k*sizeof(int8_t));
        double max_rel = 0.0, max_abs = 0.0;
        int n_bad = 0;
        for (int i = 0; i < m; ++i) {
            host_dequant_q6_k(wq + (int64_t) i*nsb, wrow, qraw, k);
            for (int j = 0; j < n_tok; ++j) {
                const block_q6_K * bx = wq + (int64_t) i*nsb;
                const block_q8_1 * by = yq + (int64_t) j*yb;
                double acc = 0.0;
                for (int sb = 0; sb < nsb; ++sb) {
                    const double d = (double) ggml_fp16_to_fp32(bx[sb].d);
                    for (int c = 0; c < 8; ++c) {
                        const double d8 = (double) ggml_fp16_to_fp32(by[sb*8 + c].d);
                        double t32 = 0.0;
                        for (int g = 0; g < 2; ++g) {
                            const int8_t * qs = by[sb*8 + c].qs;
                            const double scf = (double) bx[sb].scales[2*c + g] * d;
                            int64_t dot = 0;
                            for (int v = 0; v < 16; ++v) {
                                dot += (int64_t) qs[16*g + v] * (int64_t) qraw[sb*256 + 32*c + 16*g + v];
                            }
                            t32 += scf * (double) dot;
                        }
                        acc += d8 * t32;
                    }
                }
                const double got = dst[(int64_t) i*n_tok + j];
                const double ad  = fabs(got - acc);
                const double rel = ad / (fabs(acc) > 1e-3 ? fabs(acc) : 1e-3);
                if (rel > max_rel) max_rel = rel;
                if (ad  > max_abs) max_abs = ad;
                // fp32 accumulation of ~500-magnitude group terms rounds at ~1e-6 absolute on
                // near-zero accumulators, a mapping or scale bug shows O(1) absolute error
                if (rel > 2e-4 && ad > 2e-6) n_bad++;
                if (m <= 16 && n_tok <= 2) printf("i=%d j=%d got=%12.6f ref=%12.6f\n", i, j, got, acc);
            }
        }
        printf("check m=%d k=%d n_tok=%d: max_rel=%.3e max_abs=%.3e bad(>2e-4)=%d -> %s\n",
               m, k, n_tok, max_rel, max_abs, n_bad, n_bad ? "FAIL" : "PASS");
        free(wrow);
        free(qraw);
        return n_bad ? 1 : 0;
    }

    // bench
    for (int i = 0; i < 8; ++i) {
        mma8_q6k_matvec<<<grid, block>>>(dw, nsb, dy, ddst, m, n_tok, nsb, bmode);
    }
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const int rounds = 3;
    double best = 1e30;
    for (int r = 0; r < rounds; ++r) {
        cudaEventRecord(e0);
        for (int i = 0; i < iters; ++i) {
            mma8_q6k_matvec<<<grid, block>>>(dw, nsb, dy, ddst, m, n_tok, nsb, bmode);
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        const double t_us = ms*1e3 / iters;
        if (t_us < best) best = t_us;
    }
    const double bytes = (double) m*k*(210.0/256.0) + (double) n_tok*yb*36 + (double) m*n_tok*4;
    printf("mma8[%d] m=%d k=%d n_tok=%d: %8.2f us/iter  %6.1f GB/s over %7.2f MB weights (best of %dx%d)\n",
           bmode, m, k, n_tok, best, bytes / (best*1e-6) / 1e9, (double) m*k*210.0/256.0/1e6, rounds, iters);

    cudaFree(dw); cudaFree(dy); cudaFree(ddst);
    free(hx); free(wq); free(hact); free(yq); free(dst);
    return 0;
}
