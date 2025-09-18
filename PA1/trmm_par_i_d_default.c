void trmm_par(int N, float *__restrict__ a, float *__restrict__ b,
                 float *__restrict__ c) 
{
  #pragma omp parallel 
  {
  #pragma omp for schedule(dynamic)
  for (int i = 0; i < N; i++)
   for (int j = i; j < N; j++)
    for (int k = i; k <= j; k++)
//   c[i][j] = c[i][j] + a[i][k]*b[k][j];
     c[i*N+j]=c[i*N+j]+a[i*N+k]*b[k*N+j];
  }
}
