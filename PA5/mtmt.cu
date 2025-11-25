#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define FIXME 1

cudaEvent_t start, stop;
float tstart, elapsedTime;

void checkCUDAError(const char *msg);
// CUDA API error checking macro
#define cudaCheck(error) \
  if (error != cudaSuccess) { \
    printf("Fatal error: %s at %s:%d\n", \
      cudaGetErrorString(error), \
      __FILE__, __LINE__); \
    exit(1); \
  }

__global__ void mmkernel( float* a, float* b, float* c,
  int n) 
{
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n && j < n){
    float sum = 0.0f;
    for(int k = 0; k < n; k++){
      sum += a[k * n + i] * b[j * n + k];
    }
    c[i * n + j] = sum;
  }
}

int main(int argc, char *argv[])
{
  const int n = 1024;
  unsigned int size_A = n * n;
  float *d_A, *d_B, *d_C;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A = (float *) malloc(mem_size_A);
  unsigned int size_B = n * n;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B = (float *) malloc(mem_size_B);
  unsigned int size_C = n * n;
  unsigned int mem_size_C = size_C * sizeof(float);
  float *h_C = (float *) malloc(mem_size_C);
  float *h_Cref = (float *) malloc(mem_size_C);
  float *temp = (float *) malloc(sizeof(float)*n);
  if ((h_A == NULL) || (h_B == NULL) || (h_C == NULL) || (h_Cref == NULL)) {
    fprintf(stderr, "Failed to allocate host matrix!\n");
    exit(EXIT_FAILURE);
  }
  cudaCheck(cudaMalloc((void **) &d_A, mem_size_A));
  cudaCheck(cudaMalloc((void **) &d_B, mem_size_B));
  cudaCheck(cudaMalloc((void **) &d_C, mem_size_C));

  for (int i=0;i<n;i++) 
    for (int k=0;k<n;k++) 
      h_A[i*n+k] = rand();
  for (int j=0;j<n;j++) 
    for (int k=0;k<n;k++) 
      h_B[j*n+k] = rand();

  for (int j=0;j<n;j++)
  { for (int i=0;i<n;i++) temp[i] = 0.0;
    for (int k=0;k<n;k++)
    { float bjk = h_B[j*n+k];
      for (int i=0;i<n;i++)
        temp[i] += h_A[k*n+i] * bjk;
    }
    for (int i=0;i<n;i++) h_Cref[i*n+j] = temp[i];
  }

  cudaCheck(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));


     for (int i=0;i<n;i++) for (int j=0;j<n;j++) h_C[n*i+j] = 0.0;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     cudaEventRecord(start);
// Launch kernel
     int ty = 16; int tx = 16;
     dim3 threads(tx,ty);
     dim3 grid(ceil(n / threads.x), ceil(n / threads.y));
     mmkernel<<<grid,threads>>>(d_A, d_B, d_C, n);
     checkCUDAError("GPU kernel launch failure");
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&elapsedTime, start,stop);
     cudaDeviceSynchronize();
     cudaCheck(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    float threshold = 1e-4;
    bool error_found = false;
    for (int i = 0; i < n && !error_found; i++) {
      for (int j = 0; j < n && !error_found; j++) {
        float expected = h_Cref[i*n+j];
        float actual = h_C[i*n + j];
        if (fabs((actual-expected)/expected) > threshold) {
          printf("Error at C[%d,%d]: expected %f, got %f\n",
                 i, j, expected, actual);
          error_found = true;
        }
      }
    }
    if (!error_found) {
  printf("Matrix Size:%d, blockDim.x: %d, blockDim.y: %d; TBsize: %d; Time: %.2f msec; Performance: %.2f GFLOPs\n",n,tx,ty,tx*ty,elapsedTime,2e-6*n*n*n/elapsedTime);
    }
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

