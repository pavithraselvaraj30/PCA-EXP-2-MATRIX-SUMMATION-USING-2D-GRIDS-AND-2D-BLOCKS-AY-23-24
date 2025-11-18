# PCA-EXP-2-Matrix-Summation-using-2D-Grids-and-2D-Blocks-AY-23-24

<h3>ENTER YOUR NAME: Pavithra S</h3>
<h3>ENTER YOUR REGISTER NO: 212223230147</h3>
<h3>EX. NO: 02</h3>
<h3>DATE: 16.11.25/10/2025</h3>
<h1> <align=center> MATRIX SUMMATION WITH A 2D GRID AND 2D BLOCKS </h3>
i.  Use the file sumMatrixOnGPU-2D-grid-2D-block.cu
ii. Matrix summation with a 2D grid and 2D blocks. Adapt it to integer matrix addition. Find the best execution configuration. </h3>

## AIM:
To perform  matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1.	Initialize the data: Generate random data for two input arrays using the initialData function.
2.	Perform the sum on the host: Use the sumMatrixOnHost function to calculate the sum of the two input arrays on the host (CPU) for later verification of the GPU results.
3.	Allocate memory on the device: Allocate memory on the GPU for the two input arrays and the output array using cudaMalloc.
4.	Transfer data from the host to the device: Copy the input arrays from the host to the device using cudaMemcpy.
5.	Set up the execution configuration: Define the size of the grid and blocks. Each block contains multiple threads, and the grid contains multiple blocks. The total number of threads is equal to the size of the grid times the size of the block.
6.	Perform the sum on the device: Launch the sumMatrixOnGPU2D kernel on the GPU. This kernel function calculates the sum of the two input arrays on the device (GPU).
7.	Synchronize the device: Use cudaDeviceSynchronize to ensure that the device has finished all tasks before proceeding.
8.	Transfer data from the device to the host: Copy the output array from the device back to the host using cudaMemcpy.
9.	Check the results: Use the checkResult function to verify that the output array calculated on the GPU matches the output array calculated on the host.
10.	Free the device memory: Deallocate the memory that was previously allocated on the GPU using cudaFree.
11.	Free the host memory: Deallocate the memory that was previously allocated on the host.
12.	Reset the device: Reset the device using cudaDeviceReset to ensure that all resources are cleaned up before the program exits.

## PROGRAM:
```C++
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter


%%cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#ifndef _COMMON_H
#define _COMMON_H
#define CHECK(call) \
{ \
const cudaError_t error = call; \
if (error != cudaSuccess) \
{ \
fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
fprintf(stderr, "code: %d, reason: %s\n", error, \
cudaGetErrorString(error)); \
exit(1); \
} \
}
#define CHECK_CUBLAS(call) \
{ \
cublasStatus_t err; \
if ((err = (call)) != CUBLAS_STATUS_SUCCESS) \
{ \
fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__, \ __LINE__); \
exit(1); \
} \
}
#define CHECK_CURAND(call) \
{ \
curandStatus_t err; \
if ((err = (call)) != CURAND_STATUS_SUCCESS) \
{ \
fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__, \ __LINE__); \
exit(1); \
} \
}
#define CHECK_CUFFT(call) \
{ \
cufftResult err; \
if ( (err = (call)) != CUFFT_SUCCESS) \
{ \
fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__, \ __LINE__); \
exit(1); \
} \
}
#define CHECK_CUSPARSE(call) \
{ \
cusparseStatus_t err; \
if ((err = (call)) != CUSPARSE_STATUS_SUCCESS) \
{ \
fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__); \
cudaError_t cuda_err = cudaGetLastError(); \
if (cuda_err != cudaSuccess) \
{ \
fprintf(stderr, " CUDA error \"%s\" also detected\n", \
cudaGetErrorString(cuda_err)); \
} \
exit(1); \
} \
}
inline double seconds()
{
struct timeval tp;
struct timezone tzp;
int i = gettimeofday(&tp, &tzp);
return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
#endif // _COMMON_H
void initialData(float *ip, const int size)
{
int i;
for(i = 0; i < size; i++)
{
ip[i] = (float)( rand() & 0xFF ) / 10.0f;
}
}
void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
float *ia = A;
float *ib = B;
float *ic = C;
for (int iy = 0; iy < ny; iy++)
{
for (int ix = 0; ix < nx; ix++)
{
ic[ix] = ia[ix] + ib[ix];
}
ia += nx;
ib += nx;
ic += nx;
}
return;
}
void checkResult(float *hostRef, float *gpuRef, const int N) {
double epsilon = 1.0E-8;
int match = 1;
for (int i = 0; i < N; i++) {
if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
match = 0;
printf("Arrays do not match!\n");
printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
break;
}
}
if (match) printf("Arrays match.\n\n");
return;
}
// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int NX, int NY)
{
unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
unsigned int idx = iy * NX + ix;
if (ix < NX && iy < NY)
{
C[idx] = A[idx] + B[idx];
}
}
double cpuSecond() {
struct timeval tp;
gettimeofday(&tp,NULL);
return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
int main(int argc, char **argv) {
printf("%s Starting...\n", argv[0]);
// set up device
int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("Using Device %d: %s\n", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));
// set up date size of matrix
int nx = 1<<14;
int ny = 1<<14;
int nxy = nx*ny;
int nBytes = nxy * sizeof(float);
printf("Matrix size: nx %d ny %d\n",nx, ny);
// malloc host memory
float *h_A, *h_B, *hostRef, *gpuRef;
h_A = (float *)malloc(nBytes);
h_B = (float *)malloc(nBytes);
hostRef = (float *)malloc(nBytes);
gpuRef = (float *)malloc(nBytes);
// initialize data at host side
double iStart = cpuSecond();
initialData (h_A, nxy);
initialData (h_B, nxy);
double iElaps = cpuSecond() - iStart;
memset(hostRef, 0, nBytes);
memset(gpuRef, 0, nBytes);
// add matrix at host side for result checks
iStart = cpuSecond();
sumMatrixOnHost (h_A, h_B, hostRef, nx,ny);
iElaps = cpuSecond() - iStart;
// malloc device global memory
float *d_MatA, *d_MatB, *d_MatC;
cudaMalloc((void **)&d_MatA, nBytes);
cudaMalloc((void **)&d_MatB, nBytes);
cudaMalloc((void **)&d_MatC, nBytes);
// transfer data from host to device
cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);
// invoke kernel at host side
int dimx = 32;
int dimy = 32;
dim3 block(dimx, dimy);
dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
iStart = cpuSecond();
sumMatrixOnGPU2D <<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);
cudaDeviceSynchronize();
iElaps = cpuSecond() - iStart;
printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);
// copy kernel result back to host side
cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
// check device results
checkResult(hostRef, gpuRef, nxy);
// free device global memory
cudaFree(d_MatA);
cudaFree(d_MatB);
cudaFree(d_MatC);
// free host memory
free(h_A);
free(h_B);
free(hostRef);
free(gpuRef);
// reset device
cudaDeviceReset();
return (0);
}
```

## OUTPUT:
<img width="755" height="150" alt="image" src="https://github.com/user-attachments/assets/ac6477a9-521a-44c6-a536-540477e2c331" />


## RESULT:
The host took 22.50 seconds to complete it’s computation, while the GPU outperforms the host and completes the computation in 0.00 seconds. Therefore, float variables in the GPU will result in the best possible result. Thus, matrix summation using 2D grids and 2D blocks has been performed successfully.
