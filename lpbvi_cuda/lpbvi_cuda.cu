/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2015 Kyle Hollins Wray, University of Massachusetts
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of
 *  this software and associated documentation files (the "Software"), to deal in
 *  the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *  the Software, and to permit persons to whom the Software is furnished to do so,
 *  subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *  FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 *  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 *  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 *  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "lpbvi_cuda.h"

#include <stdio.h>

// This is not C++0x, unfortunately.
#define nullptr NULL

// This is determined by hardware, so what is below is a 'safe' guess. If this is
// off, the program might return 'nan' or 'inf'.
#define FLT_MAX 1e+35

__global__ void lpbvi_update(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
		const bool *d_A, const float *d_B,
		float gamma, unsigned int horizon,
		float *Gamma, unsigned int *pi,
		float *GammaPrime, unsigned int *piPrime)
{

}

int lpbvi_cuda(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
		const bool *d_A, const float *d_B,
		const float *d_T, const float *d_O, const float *d_R,
		float Rmin, float Rmax, float gamma, unsigned int horizon,
		unsigned int numThreads,
		float *Gamma, unsigned int *pi)
{
	// The device pointers for the alpha vectors: Gamma and GammaPrime.
	float *d_Gamma;
	float *d_GammaPrime;

	// The device pointers for the actions taken on each alpha vector: pi and piPrime.
	unsigned int *d_pi;
	unsigned int *d_piPrime;

	// Ensure the data is valid.
	if (n == 0 || m == 0 || z == 0 || r == 0 ||
			d_A == nullptr || d_B == nullptr ||
			d_T == nullptr || d_O == nullptr || d_R == nullptr ||
			Rmin >= Rmax || gamma < 0.0 || gamma >= 1.0 || horizon < 1) {
		fprintf(stderr, "Error[lpbvi_cuda]: %s", "Invalid arguments.");
		return -1;
	}

	// Also ensure that there are enough blocks and threads to run the program.
	if (numThreads % 32 != 0 || numThreads < r) {
		fprintf(stderr, "Error[lpbvi_cuda]: %s", "Invalid number of threads.");
		return -2;
	}

	// Initialize the first set of Gamma to zero.

	// For each of the updates, run PBVI.
	for (int t = 0; t < horizon; t++) {
		// Execute a kernel for the first three stages of for-loops: B, A, Z, as a 3d-block,
		// and the 4th stage for-loop over Gamma as the threads.
		if (t % 2 == 0) {
			lpbvi_update<<< dim3(r, m, z), r >>>();
		} else {
			lpbvi_update<<< dim3(r, m, z), r >>>();
		}
	}

	// Copy the final result of Gamma and pi to the variables. This assumes
	// that the memory has been allocated.

	// Free the device-side Gamma and pi.

	return 0;
}

int lpbvi_initialize_actions(unsigned int m, unsigned int r, const float *A, float *&d_A)
{
	// Ensure the data is valid.
	if (m == 0 || r == 0 || A == nullptr) {
		return -1;
	}

	// Allocate the memory on the device.
	if (cudaMalloc(&d_A, r * m * sizeof(bool)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_actions]: %s",
				"Failed to allocate device-side memory for the actions.");
		return -3;
	}

	// Copy the data from the host to the device.
	if (cudaMemcpy(d_A, A, r * m * sizeof(bool), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_actions]: %s",
				"Failed to copy memory from host to device for the actions.");
		return -3;
	}

	return 0;
}

int lpbvi_initialize_belief_points(unsigned int n, unsigned int r, const float *B, float *&d_B)
{
	// Ensure the data is valid.
	if (n == 0 || r == 0 || B == nullptr) {
		return -1;
	}

	// Allocate the memory on the device.
	if (cudaMalloc(&d_B, r * n * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_belief_points]: %s",
				"Failed to allocate device-side memory for the belief points.");
		return -3;
	}

	// Copy the data from the host to the device.
	if (cudaMemcpy(d_B, B, r * n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_belief_points]: %s",
				"Failed to copy memory from host to device for the belief points.");
		return -3;
	}

	return 0;
}

int lpbvi_initialize_state_transitions(unsigned int n, unsigned int m, const float *T, float *&d_T)
{
	// Ensure the data is valid.
	if (n == 0 || m == 0 || T == nullptr) {
		return -1;
	}

	// Allocate the memory on the device.
	if (cudaMalloc(&d_T, n * m * n * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_state_transitions]: %s",
				"Failed to allocate device-side memory for the state transitions.");
		return -3;
	}

	// Copy the data from the host to the device.
	if (cudaMemcpy(d_T, T, n * m * n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_state_transitions]: %s",
				"Failed to copy memory from host to device for the state transitions.");
		return -3;
	}

	return 0;
}

int lpbvi_initialize_observation_transitions(unsigned int n, unsigned int m, unsigned int z,
		const float *O, float *&d_O)
{
	// Ensure the data is valid.
	if (n == 0 || m == 0 || z == 0 || O == nullptr) {
		return -1;
	}

	// Allocate the memory on the device.
	if (cudaMalloc(&d_O, m * n * z * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_observation_transitions]: %s",
				"Failed to allocate device-side memory for the observation transitions.");
		return -3;
	}

	// Copy the data from the host to the device.
	if (cudaMemcpy(d_O, O, m * n * z * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_observation_transitions]: %s",
				"Failed to copy memory from host to device for the observation transitions.");
		return -3;
	}

	return 0;
}

int lpbvi_initialize_rewards(unsigned int n, unsigned int m, const float *R, float *&d_R)
{
	// Ensure the data is valid.
	if (n == 0 || m == 0 || R == nullptr) {
		return -1;
	}

	// Allocate the memory on the device.
	if (cudaMalloc(&d_R, n * m * n * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_rewards]: %s",
				"Failed to allocate device-side memory for the rewards.");
		return -3;
	}

	// Copy the data from the host to the device.
	if (cudaMemcpy(d_R, R, n * m * n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_rewards]: %s",
				"Failed to copy memory from host to device for the rewards.");
		return -3;
	}

	return 0;
}

int lpbvi_initialize_gamma_a_star(unsigned int n, unsigned int m, const float *GammaAStar, float *&d_GammaAStar)
{
	// Ensure the data is valid.
	if (m == 0 || GammaAStar == nullptr) {
		return -1;
	}

	// Allocate the memory on the device.
	if (cudaMalloc(&d_GammaAStar, m * n * sizeof(bool)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_gamma_a_star]: %s",
				"Failed to allocate device-side memory for Gamma_{a,*} over all a in A.");
		return -3;
	}

	// Copy the data from the host to the device.
	if (cudaMemcpy(d_GammaAStar, GammaAStar, m * n * sizeof(bool), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_gamma_a_star]: %s",
				"Failed to copy memory from host to device for Gamma_{a,*} over all a in A.");
		return -3;
	}

	return 0;
}

int lpbvi_uninitialize(bool *&d_A, float *&d_B, float *&d_T, float *&d_O, float *&d_R, float *d_Gamma, unsigned int *&d_pi)
{
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_T);
	cudaFree(d_O);
	cudaFree(d_R);
	cudaFree(d_Gamma);
	cudaFree(d_pi);

	return 0;
}
