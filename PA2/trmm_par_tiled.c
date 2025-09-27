void mm_par(int N, float *__restrict__ a, float *__restrict__ b,
                 float *__restrict__ c) 
{
    #pragma omp parallel 
    {
        #pragma omp for
        for (int i = 0; i < N; i+=32){
            for (int k = 0; k < N; k+=32){
                for (int j = 0; j < N; j+=32){
                    for (int ii = i; ii < i+32; ii++){
                        for (int kk = k; kk < k+32; kk++){
                            for (int jj = j; jj < j+32; jj++){
                                c[ii*N+jj]=c[ii*N+jj]+a[ii*N+kk]*b[kk*N+jj];
                            }
                        }
                    }
                }
            }
        }
    }
}