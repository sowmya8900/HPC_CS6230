// Use "clang -O3 -mavx512f -Rpass-missed=loop-vectorize -Rpass=loop-vectorize " to compile

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#define NTrials 5
#define NReps 10
#define N 16*1024

void vec1a(int size, int Reps, float *__restrict__ x);
void vec1b(int size, int Reps, float *__restrict__ x);
void vec1c(int size, int Reps, float *__restrict__ x);

double get_wtime()
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
    return (ts.tv_sec+1e-9*ts.tv_nsec);
}

float x[N];

int main(int argc, char *argv[]){
  double tstart,telapsed;
  
  int i,trial;
  double mint,maxt;
  
  printf("Matrix Size = %d; NTrials=%d\n",N,NTrials);
  
  for(i=0;i<N;i++)
    x[i] = 10.0*i/sqrt(N);
  
// Warmup
  vec1a(N, NReps, &x[0]);
  mint = 1e9; maxt = 0;
  for(trial=0;trial<NTrials;trial++)
  {
   tstart = get_wtime();
   vec1a(N, NReps, &x[0]);
   telapsed = get_wtime()-tstart;
   if (telapsed < mint) mint=telapsed;
   if (telapsed > maxt) maxt=telapsed;
  }
  printf("Performance (GFLOPS) for stmt 'w[i] = w[i]+1':");
  printf(" Min: %.2f; Max: %.2f\n",1.0e-9*N*NReps/maxt,1.0e-9*N*NReps/mint);

// Warmup
  vec1b(N, NReps, &x[0]);
  mint = 1e9; maxt = 0;
  for(trial=0;trial<NTrials;trial++)
  {
   tstart = get_wtime();
   vec1b(N, NReps, &x[0]);
   telapsed = get_wtime()-tstart;
   if (telapsed < mint) mint=telapsed;
   if (telapsed > maxt) maxt=telapsed;
  }
  printf("Performance (GFLOPS) for stmt 'w[i] = w[i+1]+1':");
  printf(" Min: %.2f; Max: %.2f\n",1.0e-9*N*NReps/maxt,1.0e-9*N*NReps/mint);

// Warmup
  vec1c(N, NReps, &x[0]);
  mint = 1e9; maxt = 0;
  for(trial=0;trial<NTrials;trial++)
  {
   tstart = get_wtime();
   vec1c(N, NReps, &x[0]);
   telapsed = get_wtime()-tstart;
   if (telapsed < mint) mint=telapsed;
   if (telapsed > maxt) maxt=telapsed;
  }
  printf("Performance (GFLOPS) for stmt 'w[i] = w[i-1]+1':");
  printf(" Min: %.2f; Max: %.2f\n",1.0e-9*N*NReps/maxt,1.0e-9*N*NReps/mint);
} 
