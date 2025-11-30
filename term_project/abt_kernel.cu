#define TILE 16

__global__ void abt_kernel(const float *A, const float *B, float *C,
                           int Ni, int Nj, int Nk)
{
    constexpr int BM = 16;  // rows of C per block (i dimension)
    constexpr int BN = 16;  // cols of C per block (j dimension)
    constexpr int BK = 16;  // k tile size

    int ty = threadIdx.y;           // 0..BM-1
    int tx = threadIdx.x;           // 0..BN-1
    int i0 = blockIdx.y * BM;       // row tile start
    int j0 = blockIdx.x * BN;       // col tile start
    int i  = i0 + ty;
    int j  = j0 + tx;

    // Shared memory with padding to avoid bank conflicts
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BN][BK + 1];

    float acc = 0.0f;

    for (int kb = 0; kb < Nk; kb += BK) {
        // Load A tile: A[i, kb + tx]
        // Each thread loads one element from A
        int kA = kb + tx;
        if (i < Ni && kA < Nk) {
            As[ty][tx] = A[i * Nk + kA];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile: B[j, kb + ty]
        // For B^T multiplication, I need B[j, k] for all k
        // Thread (tx, ty) should load B[j0+tx, kb+ty]
        int kB = kb + ty;
        int j_load = j0 + tx;
        if (j_load < Nj && kB < Nk) {
            Bs[tx][ty] = B[j_load * Nk + kB];
        } else {
            Bs[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            acc += As[ty][kk] * Bs[tx][kk];
        }

        __syncthreads();
    }

    // Write result
    if (i < Ni && j < Nj) {
        C[i * Nj + j] = acc;
    }
}