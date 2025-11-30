__global__ void abt_kernel(const float *A, const float *B, float *C, int Ni, int Nj, int Nk);

void abt_launch(const float *d_A, const float *d_B, float *d_C,
                int Ni, int Nj, int Nk)
{
    const int BM = 16, BN = 16;
    dim3 block(BN, BM);  // (x=j, y=i)
    dim3 grid((Nj + BN - 1) / BN, (Ni + BM - 1) / BM);
    abt_kernel<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
}
