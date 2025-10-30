void tmm_ref(int N, float *__restrict__ a, float *__restrict__ b,
                 float *__restrict__ c) 
{
  for (int i = 0; i < N; i++)
   for (int k = 0; k < N; k++)
    for (int j = 0; j < N; j++)
     c[j*N+i]=c[j*N+i]+a[k*N+i]*b[k*N+j];
}
