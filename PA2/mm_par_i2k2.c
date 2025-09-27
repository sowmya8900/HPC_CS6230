void mm_par(int N, float *__restrict__ a, float *__restrict__ b,
                 float *__restrict__ c) 
{
    #pragma omp parallel 
    {
        #pragma omp for
        for (int i = 0; i < N; i+=2){
            for (int k = 0; k < N; k+=2){
                for (int j = 0; j < N; j++){
                    c[i*N+j]=c[i*N+j]+a[i*N+k]*b[k*N+j];
                    c[i*N+j]=c[i*N+j]+a[i*N+(k+1)]*b[(k+1)*N+j];
                    c[(i+1)*N+j]=c[(i+1)*N+j]+a[(i+1)*N+k]*b[k*N+j];
                    c[(i+1)*N+j]=c[(i+1)*N+j]+a[(i+1)*N+(k+1)]*b[(k+1)*N+j];
                }
            }
        }
    }
}
