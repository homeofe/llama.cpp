// mma8_probe.cu - determine the m8n8k16 s8 fragment layout on sm_75 empirically.
// A = 16x16 (values r*16+k), B = 8 tokens x 16 k (values 100 + j*16 + k).
// Builds A.x[0]/A.x[1] per candidate layouts, runs mma(), prints C.x[l] vs the CPU dot.
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mma.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

using namespace ggml_cuda_mma;

__global__ void probe(const int * a, const int * b, int * cout) {
    const int lane = threadIdx.x;
    const int c = lane & 3, r = lane >> 2;
    tile<16, 4, int> A;
    tile< 8, 4, int> B;
    // B per load_generic semantics: token r, k bytes 4*c..4*c+3
    B.x[0] = b[r*4 + c];
#if LAYOUT == 1
    // L1: A.x[0] = row r, k bytes 4*c..+3; x[1] = row r+8
    A.x[0] = a[r*4 + c];
    A.x[1] = a[(r+8)*4 + c];
#elif LAYOUT == 2
    // L2: A.x[0] = row r, k bytes c, c+4, c+8, c+12; x[1] = row r+8
    {
        const unsigned char * p0 = (const unsigned char *) (a + r*4);
        const unsigned char * p1 = (const unsigned char *) (a + (r+8)*4);
        A.x[0] = (int) p0[c] | ((int) p0[c+4] << 8) | ((int) p0[c+8] << 16) | ((int) p0[c+12] << 24);
        A.x[1] = (int) p1[c] | ((int) p1[c+4] << 8) | ((int) p1[c+8] << 16) | ((int) p1[c+12] << 24);
    }
#elif LAYOUT == 3
    // L3: A.x[0] = row r, k bytes 4*c..+3 but x[1] = row r (same, other k half)
    A.x[0] = a[r*4 + c];
    A.x[1] = a[r*4 + c];
#elif LAYOUT == 4
    // L4: x[0] = row r+8, x[1] = row r
    A.x[0] = a[(r+8)*4 + c];
    A.x[1] = a[r*4 + c];
#elif LAYOUT == 5
    // L5: rows by lane%4 quadruple: x[0] = row 4*(lane%4) + lane/16?? try row = lane%8*... rows 0..7 for lanes 0..31: row = lane/4 is L1; try row = lane%8 with byte group lane/8
    A.x[0] = a[(lane % 8)*4 + (lane / 8)];
    A.x[1] = a[((lane % 8)+8)*4 + (lane / 8)];
#endif
    tile<16, 8, int> C;
    mma(C, A, B);
    #pragma unroll
    for (int l = 0; l < 4; ++l) cout[lane*4 + l] = C.x[l];
}

int main(void) {
    // host data: a[16 rows][4 int32] with byte b of int32 j = A[r][4j+b]
    static int ha[16*4], hb[8*4];
    static signed char A[16][16], B[8][16];
    for (int r = 0; r < 16; ++r) for (int k = 0; k < 16; ++k) A[r][k] = (signed char) (r*16 + k);
    for (int j = 0; j < 8; ++j)  for (int k = 0; k < 16; ++k) B[j][k] = (signed char) (100 + j*16 + k);
    for (int r = 0; r < 16; ++r) for (int i = 0; i < 4; ++i) {
        ha[r*4 + i] = (int) (unsigned char) A[r][4*i] | ((int) (unsigned char) A[r][4*i+1] << 8)
                    | ((int) (unsigned char) A[r][4*i+2] << 16) | ((int) (unsigned char) A[r][4*i+3] << 24);
    }
    for (int j = 0; j < 8; ++j) for (int i = 0; i < 4; ++i) {
        hb[j*4 + i] = (int) (unsigned char) B[j][4*i] | ((int) (unsigned char) B[j][4*i+1] << 8)
                    | ((int) (unsigned char) B[j][4*i+2] << 16) | ((int) (unsigned char) B[j][4*i+3] << 24);
    }
    int * da, * db, * dc;
    cudaMalloc(&da, sizeof(ha)); cudaMalloc(&db, sizeof(hb)); cudaMalloc(&dc, 32*4*sizeof(int));
    cudaMemcpy(da, ha, sizeof(ha), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof(hb), cudaMemcpyHostToDevice);
    probe<<<1, 32>>>(da, db, dc);
    int hc[32*4];
    cudaMemcpy(hc, dc, sizeof(hc), cudaMemcpyDeviceToHost);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("kernel error: %s\n", cudaGetErrorString(err)); return 1; }

    int n_ok = 0;
    for (int lane = 0; lane < 32; ++lane) {
        for (int l = 0; l < 4; ++l) {
            // C(i,j) per mma.cuh: i = 8*(l/2) + lane/4, j = 2*(lane%4) + l%2
            const int i = 8*(l/2) + lane/4;
            const int j = 2*(lane%4) + (l & 1);
            int expect = 0;
            for (int k = 0; k < 16; ++k) expect += (int) A[i][k] * (int) B[j][k];
            const int got = hc[lane*4 + l];
            if (got == expect) n_ok++;
            else printf("lane %d l %d: got %d expect %d (i=%d j=%d)\n", lane, l, got, expect, i, j);
        }
    }
    printf("LAYOUT %d: %d/128 fragment elements correct\n", LAYOUT, n_ok);
    return 0;
}
