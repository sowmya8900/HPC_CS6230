#include <stdio.h>
#include <time.h>
#define Ntrials 5
#define threshold 0.001

void checkCUDAError(const char *msg);

void abt_launch(const float *d_A, const float *d_B, float *d_C, int Ni, int Nj, int Nk);

int main(){

  cudaEvent_t start, stop;
  float elapsedTime, tmin, tmax;
  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  int i,j,k,Ni,Nj,Nk;

  printf("Specify Matrix dimension Ni, Nj, Nk: ");
  scanf("%d %d %d", &Ni,&Nj,&Nk);
  h_A = (float *) malloc(sizeof(float)*Nk*Ni);
  h_B = (float *) malloc(sizeof(float)*Nj*Nk);
  h_C = (float *) malloc(sizeof(float)*Ni*Nj);
  for (i=0; i<Ni; i++)
   for (k=0; k<Nk; k++)
    h_A[i*Nk+k] = 2*i+k;
  for (j=0; j<Nj; j++)
   for (k=0; k<Nk; k++)
    h_B[j*Nk+k] = 2*j+k;
  for (i=0; i<Ni; i++)
   for (j=0; j<Nj; j++) 
    h_C[i*Nj+j] = 0;

// Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, Nk*Ni*sizeof(float));
  cudaMalloc(&d_B, Nj*Nk*sizeof(float));
  cudaMalloc(&d_C, Ni*Nj*sizeof(float));
  checkCUDAError("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, Nk*Ni*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, Nj*Nk*sizeof(float), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy H2D failure");

  tmin = 1e9; tmax = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for(int trial=0;trial<Ntrials;trial++)
  {
   cudaEventRecord(start);
   // Launch kernel
   abt_launch(d_A, d_B, d_C, Ni, Nj, Nk);
   cudaEventRecord(stop);
   checkCUDAError("kernel launch");
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsedTime, start,stop);
   if (elapsedTime < tmin) tmin=elapsedTime;
   if (elapsedTime > tmax) tmax=elapsedTime;
   // Copy results back to host
   cudaMemcpy(h_C, d_C, Ni*Nj*sizeof(float), cudaMemcpyDeviceToHost);
   checkCUDAError("cudaMemcpy D2H");
   for(int i=0;i<Ni;i++)
     for(int j=0;j<Nj;j++)
      { float expected = 4.0*i*j*Nk + 1.0*(i+j)*(Nk-1)*Nk + (1.0f/6)*(Nk-1)*Nk*(2*Nk-1);
        if (fabs((h_C[i*Nj+j] - expected)/expected)>threshold)
        {printf("Error: mismatch at <%d,%d>, was: %f, should be: %f\n", i,j, h_C[i*Nj+j], expected); return -1;}
      }
  }
  printf("ABt <Ni=%d,Nj=%d,Nk=%d>: Over %d trials, Min_GFLOPS: %.2f; Max_GFLOPS: %2f\n",Ni,Nj,Nk,Ntrials,2.0e-6*Ni*Nj*Nk/tmax,2.0e-6*Ni*Nj*Nk/tmin);
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}


