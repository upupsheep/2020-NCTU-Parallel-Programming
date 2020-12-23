#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>


__device__ int diverge_gpu(float c_re, float c_im, int max) {
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < max; ++i) {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__global__ void mandelKernel(int *c, float lowerX, float lowerY, float stepX, float stepY, int resX, int resY, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
	/*
	int pix_per_thread = resX * resY / (gridDim.x * blockDim.x);
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = pix_per_thread * tId;

	int i;
	for (i = offset; i < offset + pix_per_thread; i++){
		int x = i % resX;
		int y = i / resX;
		float cr = lowerX + x * stepX;
		float ci = lowerY + y * stepY;
		c[y * resX + x] = diverge_gpu(cr, ci, maxIterations);
	}

	if (gridDim.x * blockDim.x * pix_per_thread < resX * resY && tId < (resX * resY) - (blockDim.x * gridDim.x)){
		int i = blockDim.x * gridDim.x * pix_per_thread + tId;
		int x = i % resX;
		int y = i / resX;
		float cr = lowerX + x * stepX;
		float ci = lowerY + y * stepY;
		c[y * resY + x] = diverge_gpu(cr, ci, maxIterations);
	}
	*/
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int thisX = index % resX;
	int thisY = index / resX;

	//int *partc = (int *) ((char *) c + thisX * pitch);

	float x = lowerX + thisX * stepX;
	float y = lowerY + thisY * stepY;
	c[index] = diverge_gpu(x, y, maxIterations);

}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

	int size = resX * resY;

	int *h_c;
	int *d_c;
	size_t pitch;
	//h_c = (int *)malloc(size * sizeof(int));
	cudaHostAlloc(&h_c, sizeof(int)*size, cudaHostAllocMapped);
	cudaHostGetDevicePointer(&d_c, h_c, 0);
	//cudaMalloc(&d_c, size * sizeof(int));
	cudaMallocPitch((void **)&d_c, &pitch, sizeof(int)*resX, resY);
	//cudaMemcpy2D(d_c, pitch, h_c, sizeof(int)*resX, sizeof(int)*resX, resY, cudaMemcpyHostToDevice);

	int block_size = 1024;
	int grid_size = 1;

	//dim3 dimBlock(block_size, block_size);
	//dim3 dimGrid(grid_size, grid_size);

	mandelKernel<<<size/block_size, block_size>>>(d_c, lowerX, lowerY, stepX, stepY, resX, resY, maxIterations);

	cudaDeviceSynchronize();

	//cudaMemcpy2D(h_c, sizeof(int)*resX, d_c, pitch, sizeof(int)*resX, resY, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_c, d_c, size*sizeof(int), cudaMemcpyDeviceToHost);

	memcpy(img, h_c, size * sizeof(int));

	cudaFree(d_c);
	//free(h_c);
	cudaFreeHost(h_c);
}