void tmm_opt(int N, float *__restrict__ a, float *__restrict__ b, float *__restrict__ c){
    for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++)
            for (int i = 0; i < N; i++)
                c[j*N + i] += a[k*N + i] * b[k*N + j];
}
