/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include <stdio.h>
#include "utils.h"
#include "timer.h"

enum find_min_max_type {
	neighbored, neighbored_less_divergence, interleaved
};

__global__ void do_find_min_max_neighbored(float* d_inMin, float* d_inMax, float* d_outMin, float* d_outMax, unsigned int n)
{
	const unsigned int tid = threadIdx.x;
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// boundary check
	if (idx >= n) return;


	// convert global data pointer to the local pointer of this block
	float *inMinData = d_inMin + blockIdx.x * blockDim.x;
	float *inMaxData = d_inMax + blockIdx.x * blockDim.x;

	// in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride)) == 0)
		{
			if(inMinData[tid] > inMinData[tid + stride])
			{
				inMinData[tid] = inMinData[tid + stride];
			}

			if(inMaxData[tid] < inMaxData[tid + stride])
			{
				inMaxData[tid] = inMaxData[tid + stride];
			}
		}

		// synchronize within threadblock
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		d_outMin[blockIdx.x] = inMinData[0];
		d_outMax[blockIdx.x] = inMaxData[0];
	}
}


__global__ void do_find_min_max_neighbored_less_divergence(float* d_inMin, float* d_inMax, float* d_outMin, float* d_outMax, unsigned int n)
{
	const unsigned int tid = threadIdx.x;
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// boundary check
	if (idx >= n) return;


	// convert global data pointer to the local pointer of this block
	float *inMinData = d_inMin + blockIdx.x * blockDim.x;
	float *inMaxData = d_inMax + blockIdx.x * blockDim.x;

	// in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		// convert tid into local array index
		int index = 2 * stride * tid;

		if (index < blockDim.x)
		{
			if(inMinData[index] > inMinData[index + stride])
			{
				inMinData[index] = inMinData[index + stride];
			}

			if(inMaxData[index] < inMaxData[index + stride])
			{
				inMaxData[index] = inMaxData[index + stride];
			}
		}

		// synchronize within threadblock
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		d_outMin[blockIdx.x] = inMinData[0];
		d_outMax[blockIdx.x] = inMaxData[0];
	}
}

__global__ void do_find_min_max_interleaved(float* d_inMin, float* d_inMax, float* d_outMin, float* d_outMax, unsigned int n)
{
	const unsigned int tid = threadIdx.x;
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// boundary check
	if (idx >= n) return;

	// convert global data pointer to the local pointer of this block
	float *inMinData = d_inMin + blockIdx.x * blockDim.x;
	float *inMaxData = d_inMax + blockIdx.x * blockDim.x;

	// in-place reduction in global memory
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			if(inMinData[tid] > inMinData[tid + stride])
			{
				inMinData[tid] = inMinData[tid + stride];
			}

			if(inMaxData[tid] < inMaxData[tid + stride])
			{
				inMaxData[tid] = inMaxData[tid + stride];
			}
		}

		// when stride is 3, the last step (stride=1) skips 2-th element
		if (stride == 3)
		{
			stride = 4;
		}

		// synchronize within threadblock
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		d_outMin[blockIdx.x] = inMinData[0];
		d_outMax[blockIdx.x] = inMaxData[0];
	}
}

void find_min_max(const float* d_logLuminance, size_t numPixels, float &min, float &max, find_min_max_type type)
{
	GpuTimer timer;

	size_t ARRAY_BYTES = sizeof(float) * numPixels;

	//set block/grid size
	const dim3 blockSize(512);
	const dim3 gridSize((numPixels + blockSize.x - 1) / blockSize.x);

	const dim3 finalBlockSize(gridSize.x); // launch one thread for each block in prev step
	const dim3 finalGridSize(1);


	//allocate memory for min and max outputs
	float* d_inMin = NULL;
	float* d_inMax = NULL;
	float* d_interMin = NULL;
	float* d_interMax = NULL;

	float* d_outMin = NULL;
	float* d_outMax = NULL;

	checkCudaErrors(cudaMalloc(&d_inMin, ARRAY_BYTES));
	checkCudaErrors(cudaMemcpy(d_inMin, d_logLuminance, ARRAY_BYTES, cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMalloc(&d_inMax, ARRAY_BYTES));
	checkCudaErrors(cudaMemcpy(d_inMax, d_logLuminance, ARRAY_BYTES, cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMalloc(&d_interMin, ARRAY_BYTES));
	checkCudaErrors(cudaMalloc(&d_interMax, ARRAY_BYTES));

	checkCudaErrors(cudaMalloc(&d_outMin, sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_outMax, sizeof(float)));

	checkCudaErrors(cudaMemset(d_interMin, 0, ARRAY_BYTES));
	checkCudaErrors(cudaMemset(d_interMax, 0, ARRAY_BYTES));


	timer.Start();
	switch(type)
	{
		case neighbored:
			do_find_min_max_neighbored<<<gridSize, blockSize>>>(d_inMin, d_inMax, d_interMin, d_interMax, numPixels);
			do_find_min_max_neighbored<<<finalGridSize, finalBlockSize>>>(d_interMin, d_interMax, d_outMin, d_outMax, finalBlockSize.x);
			break;
		case neighbored_less_divergence:
			do_find_min_max_neighbored_less_divergence<<<gridSize, blockSize>>>(d_inMin, d_inMax, d_interMin, d_interMax, numPixels);
			do_find_min_max_neighbored_less_divergence<<<finalGridSize, finalBlockSize>>>(d_interMin, d_interMax, d_outMin, d_outMax, finalBlockSize.x);
			break;
		case interleaved:
			do_find_min_max_interleaved<<<gridSize, blockSize>>>(d_inMin, d_inMax, d_interMin, d_interMax, numPixels);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			do_find_min_max_interleaved<<<finalGridSize, finalBlockSize>>>(d_interMin, d_interMax, d_outMin, d_outMax, finalBlockSize.x);
			break;
	}
	timer.Stop();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// copy back the min and max values from GPU
	checkCudaErrors(cudaMemcpy(&min, d_outMin, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max, d_outMax, sizeof(float), cudaMemcpyDeviceToHost));

	printf("find_min_max elapsed %f ms, min=%f, max=%f.\n", timer.Elapsed(), min, max);


	// clean up memory
	checkCudaErrors(cudaFree(d_inMin));
	checkCudaErrors(cudaFree(d_inMax));
	checkCudaErrors(cudaFree(d_interMin));
	checkCudaErrors(cudaFree(d_interMax));
	checkCudaErrors(cudaFree(d_outMin));
	checkCudaErrors(cudaFree(d_outMax));

}


__global__ void do_histogram_simple(const float *d_in, float* d_bins,
		float logLumMin, float logLumRange,
		const int numBins)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float item = d_in[idx];
	unsigned int bin =  static_cast<unsigned int>((item - logLumMin) / logLumRange * numBins);
	if(bin > numBins - 1)
		bin = numBins - 1;
	atomicAdd(&(d_bins[bin]), 1);
}

void histogram( const float* d_logLuminance, float** d_histogram,
				float logLumMin, float logLumMax, float logLumRange,
				const size_t numPixels, const size_t numBins)
{
	checkCudaErrors(cudaMalloc(d_histogram, sizeof(float)*numBins));
	checkCudaErrors(cudaMemset(*d_histogram, 0, sizeof(float)*numBins));

	const dim3 blockSize(32);
	const dim3 gridSize((numPixels + blockSize.x - 1) / blockSize.x);

	GpuTimer timer;
	timer.Start();
	do_histogram_simple<<<gridSize, blockSize>>>(d_logLuminance, *d_histogram, logLumMin, logLumRange, numBins);
	timer.Stop();
	printf("do_histogram_simple elapsed %f ms\n", timer.Elapsed());
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &logLumMin,
                                  float &logLumMax,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

	const size_t numPixels = numRows * numCols;

	// 1) find the minimum and maximum value in the input logLuminance channel store in min_logLum and max_logLum
	//find_min_max(d_logLuminance, numPixels, logLumMin, logLumMax, neighbored);
	//find_min_max(d_logLuminance, numPixels, logLumMin, logLumMax, neighbored_less_divergence);
	find_min_max(d_logLuminance, numPixels, logLumMin, logLumMax, interleaved);

	// 2) subtract them to find the range
	float logLumRange = logLumMax - logLumMin;

	// 3) generate a histogram of all the values in the logLuminance channel using the formula:
    // bin = (lum[i] - lumMin) / lumRange * numBins
	float* d_histogram = NULL;
	histogram(d_logLuminance, &d_histogram, logLumMin, logLumMax, logLumRange, numPixels, numBins);


	float* h_histogram = new float[numBins];
	checkCudaErrors(cudaMemcpy(h_histogram, d_histogram, sizeof(float)*numBins, cudaMemcpyDeviceToHost));


	// 5) Clean up memory
	checkCudaErrors(cudaFree(d_histogram));
}
