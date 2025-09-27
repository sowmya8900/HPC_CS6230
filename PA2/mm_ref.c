void mm_ref(int N, float *__restrict__ a, float *__restrict__ b,
                 float *__restrict__ c) 
{
  for (int i = 0; i < N; i++)
   for (int k = 0; k < N; k++)
    for (int j = 0; j < N; j++)
     c[i*N+j]=c[i*N+j]+a[i*N+k]*b[k*N+j];
}
