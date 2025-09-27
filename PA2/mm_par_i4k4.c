void mm_par(int N, float *__restrict__ a, float *__restrict__ b,
                 float *__restrict__ c) 
{
    #pragma omp parallel 
    {
        #pragma omp for
        for (int i = 0; i < N; i+=4){
            for (int k = 0; k < N; k+=4){
                for (int j = 0; j < N; j++){
                    // i = i
                    c[i*N + j]     += a[i*N + k]     * b[k*N + j];
                    c[i*N + j]     += a[i*N + k + 1] * b[(k + 1)*N + j];
                    c[i*N + j]     += a[i*N + k + 2] * b[(k + 2)*N + j];
                    c[i*N + j]     += a[i*N + k + 3] * b[(k + 3)*N + j];

                    // i = i+1
                    c[(i + 1)*N + j] += a[(i + 1)*N + k]     * b[k*N + j];
                    c[(i + 1)*N + j] += a[(i + 1)*N + k + 1] * b[(k + 1)*N + j];
                    c[(i + 1)*N + j] += a[(i + 1)*N + k + 2] * b[(k + 2)*N + j];
                    c[(i + 1)*N + j] += a[(i + 1)*N + k + 3] * b[(k + 3)*N + j];

                    // i = i+2
                    c[(i + 2)*N + j] += a[(i + 2)*N + k]     * b[k*N + j];
                    c[(i + 2)*N + j] += a[(i + 2)*N + k + 1] * b[(k + 1)*N + j];
                    c[(i + 2)*N + j] += a[(i + 2)*N + k + 2] * b[(k + 2)*N + j];
                    c[(i + 2)*N + j] += a[(i + 2)*N + k + 3] * b[(k + 3)*N + j];

                    // i = i+3
                    c[(i + 3)*N + j] += a[(i + 3)*N + k]     * b[k*N + j];
                    c[(i + 3)*N + j] += a[(i + 3)*N + k + 1] * b[(k + 1)*N + j];
                    c[(i + 3)*N + j] += a[(i + 3)*N + k + 2] * b[(k + 2)*N + j];
                    c[(i + 3)*N + j] += a[(i + 3)*N + k + 3] * b[(k + 3)*N + j];
                }
            }
        }
    }
}
