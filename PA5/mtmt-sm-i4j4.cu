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

__global__ void mmkernel( float* a, float* b, float* c, int n) {
    int j = blockIdx.y * 64 + threadIdx.y;
    int i = blockIdx.x * 64 + threadIdx.x;
    
    __shared__ float abuf[16][64];
    __shared__ float bbuf[64][16];
    
    float sum[4][4] = {0};
    
    for(int ks = 0; ks < n; ks += 16){
        // Load A^T with bounds checking
        abuf[threadIdx.y][threadIdx.x] = ((ks + threadIdx.y) < n && i < n) ? 
            a[(ks + threadIdx.y) * n + i] : 0.0f;
        abuf[threadIdx.y][threadIdx.x + 16] = ((ks + threadIdx.y) < n && (i + 16) < n) ? 
            a[(ks + threadIdx.y) * n + (i + 16)] : 0.0f;
        abuf[threadIdx.y][threadIdx.x + 32] = ((ks + threadIdx.y) < n && (i + 32) < n) ? 
            a[(ks + threadIdx.y) * n + (i + 32)] : 0.0f;
        abuf[threadIdx.y][threadIdx.x + 48] = ((ks + threadIdx.y) < n && (i + 48) < n) ? 
            a[(ks + threadIdx.y) * n + (i + 48)] : 0.0f;
        
        // Load B^T with bounds checking
        bbuf[threadIdx.y][threadIdx.x] = (j < n && (ks + threadIdx.x) < n) ? 
            b[j * n + (ks + threadIdx.x)] : 0.0f;
        bbuf[threadIdx.y + 16][threadIdx.x] = ((j + 16) < n && (ks + threadIdx.x) < n) ? 
            b[(j + 16) * n + (ks + threadIdx.x)] : 0.0f;
        bbuf[threadIdx.y + 32][threadIdx.x] = ((j + 32) < n && (ks + threadIdx.x) < n) ? 
            b[(j + 32) * n + (ks + threadIdx.x)] : 0.0f;
        bbuf[threadIdx.y + 48][threadIdx.x] = ((j + 48) < n && (ks + threadIdx.x) < n) ? 
            b[(j + 48) * n + (ks + threadIdx.x)] : 0.0f;
        
        __syncthreads();
        
        // Compute 4x4 tile
        for (int k = 0; k < 16; k++) {
            float a0 = abuf[k][threadIdx.x];
            float a1 = abuf[k][threadIdx.x + 16];
            float a2 = abuf[k][threadIdx.x + 32];
            float a3 = abuf[k][threadIdx.x + 48];
            
            float b0 = bbuf[threadIdx.y][k];
            float b1 = bbuf[threadIdx.y + 16][k];
            float b2 = bbuf[threadIdx.y + 32][k];
            float b3 = bbuf[threadIdx.y + 48][k];
            
            sum[0][0] += a0*b0;
            sum[0][1] += a0*b1;
            sum[0][2] += a0*b2;
            sum[0][3] += a0*b3;
            
            sum[1][0] += a1*b0;
            sum[1][1] += a1*b1;
            sum[1][2] += a1*b2;
            sum[1][3] += a1*b3;
            
            sum[2][0] += a2*b0;
            sum[2][1] += a2*b1;
            sum[2][2] += a2*b2;
            sum[2][3] += a2*b3;
            
            sum[3][0] += a3*b0;
            sum[3][1] += a3*b1;
            sum[3][2] += a3*b2;
            sum[3][3] += a3*b3;
        }
        
        __syncthreads();
    }
    
    // Store results with bounds checking
    if (i < n && j < n)
        c[i * n + j] = sum[0][0];
    if ((i + 16) < n && j < n)
        c[(i + 16) * n + j] = sum[1][0];
    if ((i + 32) < n && j < n)
        c[(i + 32) * n + j] = sum[2][0];
    if ((i + 48) < n && j < n)
        c[(i + 48) * n + j] = sum[3][0];
    
    if (i < n && (j + 16) < n)
        c[i * n + (j + 16)] = sum[0][1];
    if ((i + 16) < n && (j + 16) < n)
        c[(i + 16) * n + (j + 16)] = sum[1][1];
    if ((i + 32) < n && (j + 16) < n)
        c[(i + 32) * n + (j + 16)] = sum[2][1];
    if ((i + 48) < n && (j + 16) < n)
        c[(i + 48) * n + (j + 16)] = sum[3][1];
    
    if (i < n && (j + 32) < n)
        c[i * n + (j + 32)] = sum[0][2];
    if ((i + 16) < n && (j + 32) < n)
        c[(i + 16) * n + (j + 32)] = sum[1][2];
    if ((i + 32) < n && (j + 32) < n)
        c[(i + 32) * n + (j + 32)] = sum[2][2];
    if ((i + 48) < n && (j + 32) < n)
        c[(i + 48) * n + (j + 32)] = sum[3][2];
    
    if (i < n && (j + 48) < n)
        c[i * n + (j + 48)] = sum[0][3];
    if ((i + 16) < n && (j + 48) < n)
        c[(i + 16) * n + (j + 48)] = sum[1][3];
    if ((i + 32) < n && (j + 48) < n)
        c[(i + 32) * n + (j + 48)] = sum[2][3];
    if ((i + 48) < n && (j + 48) < n)
        c[(i + 48) * n + (j + 48)] = sum[3][3];
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
     dim3 grid((n + 63) / 64, (n + 63) / 64);
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

