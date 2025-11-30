void abt_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk) {
     // Tiling for better cache reuse, SIMD vectorization
     int tile_size_j;
     if (Nj >= 16384)       tile_size_j = 256;
     else if (Nj >= 2048)   tile_size_j = 128;
     else if (Nj >= 256)    tile_size_j = 64;
     else                   tile_size_j = 32;

     #pragma omp parallel for collapse(2) schedule(static)
     for (int i = 0; i < Ni; i++){
          for (int jj = 0; jj < Nj; jj += tile_size_j){
               int j_end = (jj + tile_size_j < Nj) ? (jj + tile_size_j) : Nj;
               for (int j = jj; j < j_end; j++){
                    double sum = 0;
                    #pragma omp simd reduction(+:sum)
                    for (int k = 0; k < Nk; k++){
                         sum += A[i*Nk+k] * B[j*Nk+k];
                    }
                    C[i*Nj+j] += sum;
               }
          }
     }
}    