// Use "clang -O3 -mavx512f -Rpass-missed=loop-vectorize -Rpass=loop-vectorize " to compile

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#ifndef N
#define N (256)
#endif
#define NTrials (5)
#define threshold (0.0000001)

int compare(int n, float wref[][n], float w[][n]);
void tmm_ref(int n, float *__restrict__ a, float *__restrict__ b, float *__restrict__c);
void tmm_opt(int n, float *__restrict__ a, float *__restrict__ b, float *__restrict__c);

double get_wtime()
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
    return (ts.tv_sec+1e-9*ts.tv_nsec);
}

float c[N][N], b[N][N], a[N][N], cc[N][N];

int main(int argc, char *argv[]){
  double tstart,telapsed;
  
  int i,j,k,trial;
  double mint_opt,maxt_opt;
  double mint_ref,maxt_ref;
  
  printf("Matrix Size = %d; NTrials=%d \n",N,NTrials);

  for(i=0;i<N;i++)
   for(j=0;j<N;j++)
   { a[i][j] = 1.1*(2*i+j)/N;
     b[i][j] = 1.2*(i+2*j)/N;
   }

  
// Warmup
  tmm_ref(N, &a[0][0], &b[0][0], &c[0][0]);
  printf("Reference sequential code performance in GFLOPS");
  mint_ref = 1e9; maxt_ref = 0;
  for(trial=0;trial<NTrials;trial++)
  {
   for(i=0;i<N;i++) for(j=0;j<N;j++) c[i][j] = 0;
   tstart = get_wtime();
   tmm_ref(N, &a[0][0], &b[0][0], &c[0][0]);
   telapsed = get_wtime()-tstart;
   if (telapsed < mint_ref) mint_ref=telapsed;
   if (telapsed > maxt_ref) maxt_ref=telapsed;
  }
   printf(" Min: %.2f; Max: %.2f\n",2.0e-9*N*N*N/maxt_ref,2.0e-9*N*N*N/mint_ref);

  
// Warmup
  tmm_opt(N, &a[0][0], &b[0][0], &cc[0][0]);
  {
   mint_opt = 1e9; maxt_opt = 0;
   for (trial=0;trial<NTrials;trial++)
   {
    for(i=0;i<N;i++) for(j=0;j<N;j++) cc[i][j] = 0;
    tstart = get_wtime();
    tmm_opt(N, &a[0][0], &b[0][0], &cc[0][0]);
    telapsed = get_wtime()-tstart;
    if (telapsed < mint_opt) mint_opt=telapsed;
    if (telapsed > maxt_opt) maxt_opt=telapsed;
   }
   if (compare(N,c,cc) < 0) return(-1);
  }
  printf("Worst/Best Performance of Optimized Version (GFLOPS): "); 
  printf("%.2f/%.2f\n",2.0e-9*N*N*N/maxt_opt,2.0e-9*N*N*N/mint_opt);
}

int compare(int n, float wref[][n], float w[][n])
{
  float maxdiff,this_diff;
  int numdiffs;
  int i,j;
  numdiffs = 0;
  maxdiff = 0;
  for (i=0;i<n;i++)
    for (j=0;j<n;j++)
      {
        this_diff = wref[i][j]-w[i][j];
        if (this_diff < 0) this_diff = -1.0*this_diff;
        if (this_diff>threshold)
          { numdiffs++;
            if (this_diff > maxdiff) maxdiff=this_diff;
          }
      }
  if (numdiffs > 0)
  { printf("Error: %d Differences found over threshold %f; Max Diff = %f\n",
           numdiffs,threshold,maxdiff);
    return(-1);
  }
  else return(0);
}
