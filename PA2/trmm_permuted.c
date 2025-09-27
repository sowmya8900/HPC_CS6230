void trmm_permuted(int N, float *__restrict__ a, float *__restrict__ b,
                   float *__restrict__ c) 
{
    // Permuted loop permutation: i-k-j order
    // This permutation provides optimal cache locality:
    // - c[i*N+j]: stride-1 access in j (excellent)
    // - a[i*N+k]: same location reused for entire j-loop (excellent) 
    // - b[k*N+j]: stride-1 access in j (excellent)
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        for (int k = i; k < N; k++) {
            for (int j = k; j < N; j++) {
                c[i*N+j]=c[i*N+j]+a[i*N+k]*b[k*N+j];
            }
        }
    }
}