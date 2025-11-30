void atbt_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk) {
     // Tiling for cache efficiency, better loop nesting
     int tile_size_k;
     if (Nk >= 131072)      tile_size_k = 512;
     else if (Nk >= 8192)   tile_size_k = 256;
     else if (Nk >= 1024)   tile_size_k = 128;
     else                   tile_size_k = 64;

     int tile_size_j;
     if (Nj >= 16384)      tile_size_j = 512;
     else if (Nj >= 2048)   tile_size_j = 256;
     else if (Nj >= 256)    tile_size_j = 128;
     else                   tile_size_j = 64;

     int tile_size_i;
     if (Ni >= 16384)       tile_size_i = 64;
     else if (Ni >= 2048)   tile_size_i = 32;
     else                   tile_size_i = 16;

     #pragma omp parallel for collapse(2) schedule(static)
     for (int ii = 0; ii < Ni; ii += tile_size_i){
          int i_end = (ii + tile_size_i < Ni) ? (ii + tile_size_i) : Ni;
          for (int jj = 0; jj < Nj; jj += tile_size_j){
               int j_end = (jj + tile_size_j < Nj) ? (jj + tile_size_j) : Nj;
               for (int kk = 0; kk < Nk; kk += tile_size_k){
                    int k_end = (kk + tile_size_k < Nk) ? (kk + tile_size_k) : Nk;
                    for(int i = ii; i < i_end; i++){
                         for(int k = kk; k < k_end; k++){
                              double a_val = A[k*Ni+i];
                              #pragma omp simd
                              for (int j = jj; j < j_end; j++){
                                   C[i*Nj+j] += a_val * B[j*Nk+k];
                              }
                         }
                    }
               }
          }
     }
}

