#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>
#include <stdlib.h>
// #define N 32
// #define T 4*256*256
#define N 8192
#define T 4


void walltime(double *s,double *mus)
{ struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  (*s) = Tp.tv_sec;
  (*mus) = Tp.tv_usec;
}

int main(){

int i,j,it;
double Sec1,Sec2,MUS1,MUS2,Time;
float *A;

A = (float *) malloc(sizeof(float)*N*N);
  for (j=0; j<N; j++)
      for (i=0; i<N; i++)
            A[i*N+j] = 0.5*(i+j)/N;
  walltime(&Sec1,&MUS1);

for(it=0; it<T; it++)
  for (j=0; j<N; j++)
      for (i=0; i<N; i++)
            A[i*N+j] += it;

  walltime(&Sec2,&MUS2);
Time = (Sec2-Sec1+1.0E-6*(MUS2-MUS1));
  
  if (A[(N/2)*N + N/2] < -1) printf("Bug - Should not get here!!! %f",A[(N/2)*N + N/2]);
  printf("Matrix Dimension: %d, Repeats: %d; Time=%.2f, GFLOPS= %.2f\n",N,T,Time,
         1.0E-9*N*N*T/Time);

}

