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
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))
#define TILE_WIDTH 16



__global__ void convolution(float *I, const float *M,
	float *P, int channels, int width, int height) {

	int col = blockIdx.x * blockDim.x + threadIdx.x; 
	int row = blockIdx.y * blockDim.y + threadIdx.y; 
	
	float sum = 0;

	if (row < height && col < width ){ 
		for (int i = 0; i < channels; i++){
			
			int startY = col + Mask_radius;
			int startX = row + Mask_radius;

			for (int j = 0; j < Mask_width; j++){ 
				for (int k = 0; k < Mask_width; k++){ 

					int x = startX - j;
					int y = startY - k;

					if (y >= -1 && y < width && x >= -1 && x < height){
						float pixel = I[(x * width + y) * channels + i];
						float mask = M[j * Mask_width + k];
						sum += pixel * mask;

					}
				}
			}
			P[(row*width + col) * channels + i] = clamp(sum);
		}
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
	imageChannels = wbImage_getChannels(inputImage); //Should be three?
	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);


	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");

	cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));

	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	//TODO: INSERT CODE HERE // Done
	cudaMemcpy(deviceInputImageData, hostInputImageData,
                imageWidth * imageHeight * imageChannels * sizeof(float),
                cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData, hostMaskData,
               maskRows * maskColumns * sizeof(float),
                cudaMemcpyHostToDevice);

	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");
	//TODO: INSERT CODE HERE //Done

	//dim3 dimBlock(32, 32);
	//dim3 dimGrid(16, 16);

	dim3 dimGrid(ceil((float)imageWidth / TILE_WIDTH), ceil((float)imageHeight / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData,
imageChannels, imageWidth, imageHeight);

    //convolution_simple<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData,
//imageChannels, imageWidth, imageHeight);

	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Doing the computation on the GPU");

	wbTime_start(Copy, "Copying data from the GPU");
	//TODO: INSERT CODE HERE //Done
    cudaMemcpy(hostOutputImageData, deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);

	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(arg, outputImage);

	//TODO: RELEASE CUDA MEMORY /Done

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
