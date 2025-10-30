void vec2_opt(int n, float *__restrict__ A){
    int i, j;
    for(i=1; i<n; i++){
        for (j=0; j<n; j++){
            // A[i][j] = A[i-1][j]+1;
            A[i*n+j] = A[(i-1)*n+j]+1;    
        }
    }
}
