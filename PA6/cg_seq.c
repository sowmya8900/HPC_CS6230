#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

int cg_seq(double *A, double *b, double *x, int N)
{
    const double THRESHOLD = 0.000001;
    const int MAX_ITER = 1000;
    
    double *r = (double *)malloc(N * sizeof(double));
    double *p = (double *)malloc(N * sizeof(double));
    double *Ap = (double *)malloc(N * sizeof(double));
    
    // r = b - A*x
    // Compute A*x
    for (int i = 0; i < N; i++) {
        r[i] = 0.0;
        for (int j = 0; j < N; j++) {
            r[i] += A[i * N + j] * x[j];
        }
        r[i] = b[i] - r[i];
	p[i] = r[i];
    }
    
    // Compute initial r*r
    double rdotr = 0.0;
    for (int i = 0; i < N; i++) 
        rdotr += r[i] * r[i];
    
    int iter;
    for (iter = 0; iter < MAX_ITER; iter++) 
    {
        // Check convergence
        double res = sqrt(rdotr/N);
        if (res < THRESHOLD) {
            break;
        }
        
        // Compute A*p
        for (int i = 0; i < N; i++) {
            Ap[i] = 0.0;
            for (int j = 0; j < N; j++) 
                Ap[i] += A[i * N + j] * p[j];
        }
        
        // Compute alpha = r*r / (p*A*p)
        double pAp = 0.0;
        for (int i = 0; i < N; i++) 
            pAp += p[i] * Ap[i];

        double alpha = rdotr / pAp;
        
        
        // Update solution: x = x + alpha*p
        for (int i = 0; i < N; i++) 
            x[i] += alpha * p[i];
        
        // Update residual: r = r - alpha*A*p
        for (int i = 0; i < N; i++) 
            r[i] -= alpha * Ap[i];
        
        // Compute new r*r
        double rdotr_new = 0.0;
        for (int i = 0; i < N; i++) 
            rdotr_new += r[i] * r[i];
        
        // Compute beta = r_new*r_new / r_old*r_old
        double beta = rdotr_new / rdotr;
        
        // Update search direction: p = r + beta*p
        for (int i = 0; i < N; i++) 
            p[i] = r[i] + beta * p[i];
     
        rdotr = rdotr_new;
    }
    
    free(r);
    free(p);
    free(Ap);
    
    return iter;
}
