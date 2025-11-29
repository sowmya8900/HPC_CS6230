#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

int cg_par(double *A, double *b, double *x, int N)
{
    const double THRESHOLD = 1e-6;
    const int MAX_ITER = 1000;

    int myid, nprocs;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &nprocs);

    // block distribution with remainder
    int q = N / nprocs;
    int r = N % nprocs;
    int start   = myid * q + (myid < r ? myid : r);
    int local_n = q + (myid < r ? 1 : 0);
    int end     = start + local_n;

    // local arrays
    double *r_local  = (double *)calloc(local_n, sizeof(double));
    double *Ap_local = (double *)calloc(local_n, sizeof(double));

    // replicated vectors
    double *p         = (double *)calloc(N, sizeof(double));
    double *r_global  = (double *)calloc(N, sizeof(double));

    if (!r_local || !Ap_local || !p || !r_global) {
        fprintf(stderr, "Process %d: allocation failed\n", myid);
        MPI_Abort(comm, -1);
    }

    // broadcast initial x so all ranks have the same starting vector
    MPI_Bcast(x, N, MPI_DOUBLE, 0, comm);

    // initial residual: r_local = b_local - A_local*x
    for (int i = 0; i < local_n; ++i) {
        int gi = start + i;
        const double *Ai = &A[gi * (size_t)N];
        double Ax_i = 0.0;
        for (int j = 0; j < N; ++j) Ax_i += Ai[j] * x[j];
        r_local[i] = b[gi] - Ax_i;
        r_global[gi] = r_local[i]; // fill owned entries
    }

    // assemble full r on all ranks
    int *counts = (int *)malloc(nprocs * sizeof(int));
    int *displs = (int *)malloc(nprocs * sizeof(int));
    int base = N / nprocs, rem = N % nprocs, off = 0;
    for (int pidx = 0; pidx < nprocs; ++pidx) {
        counts[pidx] = base + (pidx < rem ? 1 : 0);
        displs[pidx] = off;
        off += counts[pidx];
    }
    MPI_Allgatherv(r_local, local_n, MPI_DOUBLE,
                   r_global, counts, displs, MPI_DOUBLE, comm);

    // p = r (replicated)
    for (int j = 0; j < N; ++j) p[j] = r_global[j];

    // rdotr (global)
    double rdotr_local = 0.0;
    for (int i = 0; i < local_n; ++i) rdotr_local += r_local[i] * r_local[i];
    double rdotr = 0.0;
    MPI_Allreduce(&rdotr_local, &rdotr, 1, MPI_DOUBLE, MPI_SUM, comm);

    int iter;
    for (iter = 0; iter < MAX_ITER; ++iter) {
        // convergence
        double res = sqrt(rdotr / (double)N);
        if (res < THRESHOLD) break;

        // Ap_local = A_local * p
        for (int i = 0; i < local_n; ++i) {
            int gi = start + i;
            const double *Ai = &A[gi * (size_t)N];
            double sum = 0.0;
            for (int j = 0; j < N; ++j) sum += Ai[j] * p[j];
            Ap_local[i] = sum;
        }

        // pAp (global)
        double pAp_local = 0.0;
        for (int i = 0; i < local_n; ++i) {
            int gi = start + i;
            pAp_local += p[gi] * Ap_local[i];
        }
        double pAp = 0.0;
        MPI_Allreduce(&pAp_local, &pAp, 1, MPI_DOUBLE, MPI_SUM, comm);
        if (pAp == 0.0) break;

        double alpha = rdotr / pAp;

        // x update: replicated update across all entries
        for (int j = 0; j < N; ++j) x[j] += alpha * p[j];

        // r_local update: r = r - alpha*Ap
        for (int i = 0; i < local_n; ++i) r_local[i] -= alpha * Ap_local[i];

        // rdotr_new (global)
        double rdotr_new_local = 0.0;
        for (int i = 0; i < local_n; ++i) rdotr_new_local += r_local[i] * r_local[i];
        double rdotr_new = 0.0;
        MPI_Allreduce(&rdotr_new_local, &rdotr_new, 1, MPI_DOUBLE, MPI_SUM, comm);

        double beta = rdotr_new / rdotr;

        // gather r and update p = r + beta*p (replicated)
        MPI_Allgatherv(r_local, local_n, MPI_DOUBLE,
                       r_global, counts, displs, MPI_DOUBLE, comm);
        for (int j = 0; j < N; ++j) p[j] = r_global[j] + beta * p[j];

        rdotr = rdotr_new;
    }

    // x is already identical on all ranks; no gather needed.

    free(r_local);
    free(Ap_local);
    free(p);
    free(r_global);
    free(counts);
    free(displs);

    return iter;
}
