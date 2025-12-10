void atbt_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk) {
    // Adaptive tiling based on problem dimensions
    int tile_i = (Ni >= 8192) ? 64 : (Ni >= 2048) ? 48 : 32;
    int tile_j = (Nj >= 8192) ? 256 : (Nj >= 2048) ? 192 : 128;
    int tile_k = (Nk >= 65536) ? 512 : (Nk >= 8192) ? 256 : 128;
    
    // Micro-kernel blocking for registers
    const int MR = 4;
    const int NR = 8;
    
    #pragma omp parallel
    {
        // Thread-local accumulation buffer to reduce false sharing
        double *local_C = (double*)malloc(tile_i * tile_j * sizeof(double));
        
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int ii = 0; ii < Ni; ii += tile_i) {
            for (int jj = 0; jj < Nj; jj += tile_j) {
                int i_end = (ii + tile_i < Ni) ? (ii + tile_i) : Ni;
                int j_end = (jj + tile_j < Nj) ? (jj + tile_j) : Nj;
                
                // Zero local accumulator
                int local_rows = i_end - ii;
                int local_cols = j_end - jj;
                for (int idx = 0; idx < local_rows * local_cols; idx++) {
                    local_C[idx] = 0.0;
                }
                
                // Accumulate into local buffer
                for (int kk = 0; kk < Nk; kk += tile_k) {
                    int k_end = (kk + tile_k < Nk) ? (kk + tile_k) : Nk;
                    
                    // Micro-kernel with register blocking
                    for (int i = ii; i < i_end; i += MR) {
                        for (int j = jj; j < j_end; j += NR) {
                            int i_limit = (i + MR < i_end) ? (i + MR) : i_end;
                            int j_limit = (j + NR < j_end) ? (j + NR) : j_end;
                            
                            // Register-blocked accumulation
                            double acc[4][8] = {{0}};
                            
                            for (int k = kk; k < k_end; k++) {
                                // Load A values
                                double a_val[4];
                                for (int mi = 0; mi < i_limit - i; mi++) {
                                    a_val[mi] = A[k * Ni + i + mi];
                                }
                                
                                // Vectorizable inner loop
                                #pragma omp simd
                                for (int nj = 0; nj < j_limit - j; nj++) {
                                    double b_val = B[(j + nj) * Nk + k];
                                    for (int mi = 0; mi < i_limit - i; mi++) {
                                        acc[mi][nj] += a_val[mi] * b_val;
                                    }
                                }
                            }
                            
                            // Write back to local buffer
                            for (int mi = 0; mi < i_limit - i; mi++) {
                                for (int nj = 0; nj < j_limit - j; nj++) {
                                    local_C[(i + mi - ii) * local_cols + (j + nj - jj)] += acc[mi][nj];
                                }
                            }
                        }
                    }
                }
                
                // Copy local results to global C
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        C[i * Nj + j] = local_C[(i - ii) * local_cols + (j - jj)];
                    }
                }
            }
        }
        
        free(local_C);
    }
}
