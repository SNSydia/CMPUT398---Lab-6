#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
	    }                                                                     \
    } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

__global__ void convolution(float *I, const float *M,
	float *P, int channels, int width, int height) {
	//TODO: INSERT CODE HERE


	int k, x, y;
	int dest, destY, destX, src, srcX, srcY;
	
	float sum = 0;
	

	__shared__ float N_ds[w][w];

	//Loading into shared emory
	for (k = 0; k < channels; k++) {
		dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
		destY = dest / w;
		destX = dest % w;
		srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
		srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
		src = (srcY * width + srcX) * channels + k;
		if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
			N_ds[destY][destX] = I[src];
		else
			N_ds[destY][destX] = 0;


		dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
		destY = dest / w, destX = dest % w;
		srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
		srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
		src = (srcY * width + srcX) * channels + k;
		if (destY < w) {
			if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
				N_ds[destY][destX] = I[src];
			else
				N_ds[destY][destX] = 0;
		}

		__syncthreads();

		int x, y;
sum = 0;
		for (y = 0; y < Mask_width; y++)
			
			for (x = 0; x < Mask_width; x++)
				sum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * Mask_width + x];
		y = blockIdx.y * TILE_WIDTH + threadIdx.y;
		x = blockIdx.x * TILE_WIDTH + threadIdx.x;
		if (y < height && x < width)
			P[(y * width + x) * channels + k] = clamp(sum);

		__syncthreads();
	}
}
int main(int argc, char *argv[]) {
	wbArg_t arg;
	int maskRows;
	int maskColumns;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	char *inputMaskFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *hostMaskData;
	float *deviceInputImageData;
	float *deviceOutputImageData;
	float *deviceMaskData;

	arg = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(arg, 0);
	inputMaskFile = wbArg_getInputFile(arg, 1);

	inputImage = wbImport(inputImageFile);
	hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

	assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
	assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);

	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	//TODO: INSERT CODE HERE
	cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceMaskData, maskRows * maskColumns * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));

	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	//TODO: INSERT CODE HERE
	cudaMemcpy(deviceInputImageData, hostInputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData, hostMaskData,
		maskRows * maskColumns * sizeof(float),
		cudaMemcpyHostToDevice);

	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");
	//TODO: INSERT CODE HERE
	
	//dim3 dimBlock(32, 32);
	//dim3 dimGrid(16, 16);
	
	dim3 dimGrid(ceil((float)imageWidth / TILE_WIDTH), ceil((float)imageHeight / TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

	convolution << <dimGrid, dimBlock >> >(deviceInputImageData, deviceMaskData, deviceOutputImageData,
		imageChannels, imageWidth, imageHeight);
	
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Doing the computation on the GPU");

	wbTime_start(Copy, "Copying data from the GPU");
	//TODO: INSERT CODE HERE

	cudaMemcpy(hostOutputImageData, deviceOutputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float),
		cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(arg, outputImage);

	//TODO: RELEASE CUDA MEMORY

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceMaskData);

	free(hostMaskData);
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}
