// Use "cc -O3 -fopenmp abt_main.c abt_par.c " 

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define NTrials (3)
#define threshold (0.001)

void abt_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk);

int main(int argc, char *argv[]){
  double tstart,telapsed;

  int i,j,k,nt,trial,nthreads;
  double mint_par,maxt_par,t_seq;

  double *A, *B, *C, *Cref;
  int Ni,Nj,Nk;

  printf("Enter Matrix dimensions Ni Nj Nk: ");
  scanf("%d %d %d", &Ni,&Nj,&Nk);
  A = (double *) malloc(sizeof(double)*Nk*Ni);
  B = (double *) malloc(sizeof(double)*Nj*Nk);
  C = (double *) malloc(sizeof(double)*Ni*Nj);
  Cref = (double *) malloc(sizeof(double)*Ni*Nj);
  for (i=0; i<Ni; i++)
   for (k=0; k<Nk; k++)
    A[i*Nk+k] = 2*i+k;
  for (j=0; j<Nj; j++)
   for (k=0; k<Nk; k++)
    B[j*Nk+k] = 2*j+k;
  for (i=0; i<Ni; i++)
   for (j=0; j<Nj; j++) {
    C[i*Nj+j] = 0;
    Cref[i*Nj+j] = 0;}

  for(i=0;i<Ni;i++) for(j=0;j<Nj;j++) Cref[i*Nj+j] = 0;

  nthreads = omp_get_max_threads();

  omp_set_num_threads(nthreads);
  mint_par = 1e9; maxt_par = 0;
  for (trial=0;trial<NTrials;trial++)
  {
   for(i=0;i<Ni;i++) for(j=0;j<Nj;j++) C[i*Nj+j] = 0;
   tstart = omp_get_wtime();
   abt_par(A,B,C,Ni,Nj,Nk);
   telapsed = omp_get_wtime()-tstart;
   if (telapsed < mint_par) mint_par=telapsed;
   if (telapsed > maxt_par) maxt_par=telapsed;
   for(int i=0;i<Ni;i++)
     for(int j=0;j<Nj;j++)
      { float expected = 4.0*i*j*Nk + 1.0*(i+j)*(Nk-1)*Nk + (1.0f/6)*(Nk-1)*Nk*(2*Nk-1);
        if (fabs((C[i*Nj+j] - expected)/expected)>threshold) 
        {printf("Error: mismatch at <%d,%d>, was: %f, should be: %f\n", i,j, C[i*Nj+j], expected); return -1;}
      }
  }
  printf("Ni:%d, Nj:%d, Nk:%d, #Threads:%d; ",Ni,Nj,Nk,nthreads);
  printf("Best/Worst Performance (GFLOPS): ");
  printf("%.2f, ",2.0e-9*Ni*Nj*Nk/mint_par);
  printf("%.2f ",2.0e-9*Ni*Nj*Nk/maxt_par);
  printf("\n");
}

