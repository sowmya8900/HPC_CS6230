#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>


int cg_seq(double *A, double *b, double *x, int N);
int cg_par(double *A, double *b, double *x, int N);

  int main(int argc, char *argv[]) {

  double clkbegin, clkend;
  double t, tseq, tpar;
  int myid, nprocs, iterations;
  int N = 1000;
  double *A = (double *)malloc(N*N*sizeof(double));
  double *b = (double *)malloc(N*sizeof(double));
  double *x = (double *)malloc(N*sizeof(double));

  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &myid );
  MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

  for(int i=0;i<N;i++)
  {
    x[i] = 1.0;
    b[i] = 0.0;
    A[i*N+i] = i+N;
    for(int j=0;j<N;j++)
    {
      if (j != i) A[i * N + j] = 1.0*(i+j)/(10.0*N);
      b[i] += A[i*N+j]*x[i];
    }
  }

  if (myid == 0)
  {
// Warmup
    for(int i=0;i<N;i++) x[i] = 0.0;
    iterations = cg_seq(A, b, x, N);
    for(int i=0;i<N;i++) x[i] = 0.0;
    clkbegin = MPI_Wtime();
    iterations = cg_seq(A, b, x, N);
    clkend = MPI_Wtime();
    tseq = clkend-clkbegin;

    // Verify: compute A*x and compare with b
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++)
            sum += A[i * N + j] * x[j];
        if ((fabs(b[i]-sum)/b[i]) > 0.00001)
        {printf("Error for sequential CG: Mismatch at i=%d; Expected=%f, Found=%f\n",i,b[i],sum); return(-1);}
    }
    printf("N=%d; Sequential CG passed correctness; converged in %d iterations; time taken: %f secs\n", N, iterations,tseq);
  }

  for(int i=0;i<N;i++) x[i] = 0.0;
  MPI_Barrier(MPI_COMM_WORLD);
  clkbegin = MPI_Wtime();
  iterations = cg_par(A, b, x, N);
  clkend = MPI_Wtime();
  t = clkend-clkbegin;
  MPI_Reduce(&t, &tpar, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (myid == 0)
  {
    // Verify: compute A*x and compare with b
    for (int i = 0; i < N; i++) {
        double sum = 0.0; 
        for (int j = 0; j < N; j++)
            sum += A[i * N + j] * x[j];
        if ((fabs(b[i]-sum)/b[i]) > 0.00001)
        {printf("Error for parallel CG: Mismatch at i=%d; Expected=%f, Found=%f\n",i,b[i],sum); return(-1);}
    }
    printf("N=%d; Parallel CG passed correctness; converged in %d iterations; time taken: %f secs\n", N, iterations,tpar);

    printf("Speedup from parallel execution on %d processes:%f\n",nprocs, tseq/tpar);
  }
  MPI_Finalize();
}

