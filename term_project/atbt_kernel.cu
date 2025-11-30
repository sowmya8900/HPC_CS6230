#define TILE 16

__global__ void atbt_kernel(const float *A, const float *B, float *C,
                            int Ni, int Nj, int Nk)
{
    constexpr int BM = 16;
    constexpr int BN = 16;
    constexpr int BK = 16;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int i0 = blockIdx.y * BM;
    int j0 = blockIdx.x * BN;
    int i  = i0 + ty;
    int j  = j0 + tx;

    // Shared memory with padding to avoid bank conflicts
    __shared__ float As[BM][BK + 1];   
    __shared__ float Bs[BN][BK + 1];   

    float acc = 0.0f;

    for (int kb = 0; kb < Nk; kb += BK) {
        // Load A tile: A^T means A is stored as [Nk x Ni], access A[k, i]
        // Thread (tx, ty) loads A[kb+ty, i0+tx] into As[tx][ty]
        // This gives coalesced access when threads with consecutive tx load
        int kA = kb + ty;
        int i_load = i0 + tx;
        if (kA < Nk && i_load < Ni) {
            As[tx][ty] = A[kA * Ni + i_load];
        } else {
            As[tx][ty] = 0.0f;
        }

        // Load B tile: B is [Nj x Nk], I need B[j, k]
        // Thread (tx, ty) loads B[j0+tx, kb+ty]
        int kB = kb + ty;
        int j_load = j0 + tx;
        if (j_load < Nj && kB < Nk) {
            Bs[tx][ty] = B[j_load * Nk + kB];
        } else {
            Bs[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Compute: C[i,j] = sum_k A[k,i] * B[j,k]
        // As[ty][kk] has A[kb+kk, i0+ty] = A[k, i]
        // Bs[tx][kk] has B[j0+tx, kb+kk] = B[j, k]
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            acc += As[ty][kk] * Bs[tx][kk];
        }

        __syncthreads();
    }

    if (i < Ni && j < Nj) {
        C[i * Nj + j] = acc;
    }
}