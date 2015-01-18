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
#define FLT_MIN -1e+35

//__global__ void lpbvi_update_belief_state_dot_product_step(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
//		const bool *A, const float *B, const float *T, const float *O, const float *R, float gamma,
//		const float *Gamma, const unsigned int *pi, float *GammaPrime, unsigned int *piPrime,
//		unsigned int beliefIndex, unsigned int a, unsigned int o,
//		float *maxAlphaDotBeta)
//{
//	// Each block is for a particular action, observation, and alpha-vector in Gamma_{b, a, omega} (which are Gamma^{t-1}).
//	unsigned int alphaIndex = blockIdx.x;
//
//	// Each thread is over states, and they stride if needed.
//	for (unsigned int s = threadIdx.x; s < n; s += blockDim.x) {
//		// We compute the value of this state in the alpha-vector, then multiply it by the belief, and add it to
//		// the current dot product value for this alpha-vector.
//		double value = 0.0;
//		for (unsigned int sp = 0; sp < n; sp++) {
//			value += T[s * m * n + a * n + sp] * O[a * n * z + sp * z + o] * Gamma[alphaIndex * n + sp];
//		}
//		value *= gamma * B[beliefIndex * n + s];
//		maxAlphaDotBeta[alphaIndex] += value;
//	}
//}
//
//// We do a reduction to compute the max index within maxAlphaDotBeta.
//__global__ void lpbvi_update_belief_state_max_step(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
//		const bool *A, const float *B, const float *T, const float *O, const float *R, float gamma,
//		const float *Gamma, const unsigned int *pi, float *GammaPrime, unsigned int *piPrime,
//		float *belief, unsigned int action, unsigned int observation,
//		float *maxAlphaDotBeta)
//{
//
//}
//
//// We compute the alpha-vector of maxAlphaDotBeta, but instead of storing it, we add it to the alphaBAStar.
//__global__ void lpbvi_update_belief_state_max_alpha_vector_step(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
//		const bool *A, const float *B, const float *T, const float *O, const float *R, float gamma,
//		const float *Gamma, const unsigned int *pi, float *GammaPrime, unsigned int *piPrime,
//		float *belief, unsigned int action, unsigned int observation, unsigned int maxAlphaIndex,
//		float *alphaBAStar)
//{
//
//}

// Find the max of each alphaBAStar using a reduction over all actions. Remember to set the action index in pi as part of this.

// For all the belief points, we execute independent code which computes the next alpha-vector to replace it (but in GammaPrime),
// using the current set of alpha-vectors (in Gamma).


__global__ void lpbvi_update(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
		const bool *A, const float *B, const float *T, const float *O, const float *R, float gamma,
		const float *Gamma, const unsigned int *pi,
		float *alphaBA,
		float *GammaPrime, unsigned int *piPrime)
{
	// For each beliefIndex, we will store an alpha-vector of size n. Hence, this must be of
	// size r * n. This is used to hold intermediate values while trying to find the maximal
	// action.
//	extern __shared__ float alphaBA[];

	// Each block will run a different belief. Our overall goal: Compute the value
	// of GammaPrime[beliefIndex * n + ???] and piPrime[beliefIndex].
	unsigned int beliefIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (beliefIndex >= r) {
		return;
	}

	// Each thread deals with a different action-observation pair. Since memory can be
	// shared among threads in a block, we can easily store results in shared memory to
	// compute the maximal alpha-vector over actions.
//	unsigned int action = blockIdx.y;
//	unsigned int observation = blockIdx.z;

	// We want to compute: dot(alphaBA, belief). Instead of storing all these intermediate
	// alpha-vectors, we will just store the values of the dot products. Once we figure out
	// which action yields the largest value, then we'll compute the actual alpha-vector and
	// store it in GammaPrime[beliefIndex * n + ???], as well as set the action piPrime[beliefIndex].

	// Now we know the maximal alpha-vector for this action and observation. Compute the actual value
	// of this observation by summing each state in the alpha-vector to form a new one, plus the original
	// GammaAStar value. Since we will dot-product this with the belief, we will just do that here.

	// We want to find the action that maximizes the value, store it in piPrime, as well as its alpha-vector GammaPrime.
	float maxActionValue = FLT_MIN;

	for (unsigned int action = 0; action < m; action++) {
		// Only execute if the action is available.
		if (A[beliefIndex * m + action]) {
			// Compute Gamma_{a,*} and set it to the first value of alphaBA.
			for (unsigned int s = 0; s < n; s++) {
				alphaBA[beliefIndex * n + s] = R[s * m + action];
			}

			// Since the bottleneck is almost always read access to global memory, write access is fine here. We will
			// overwrite old alpha-vector values if this iteration is better than previous ones.
			for (unsigned int observation = 0; observation < z; observation++) {
				// Compute the max alpha vector from Gamma, given the fixed action and observation.
				float maxAlphaDotBeta = 0.0f;
				unsigned int maxAlphaIndex = 0;

				for (unsigned int alphaIndex = 0; alphaIndex < r; alphaIndex++) {
					float alphaDotBeta = 0.0f;

					for (unsigned int s = 0; s < n; s++) {
						// We compute the value of this state in the alpha-vector, then multiply it by the belief, and add it to
						// the current dot product value for this alpha-vector.
						float value = 0.0f;
						for (unsigned int sp = 0; sp < n; sp++) {
							value += T[s * m * n + action * n + sp] * O[action * n * z + sp * z + observation] * Gamma[alphaIndex * n + sp];
						}
						alphaDotBeta += gamma * value * B[beliefIndex * n + s];
					}

					// Store the maximal value and index.
					if (alphaIndex == 0 || alphaDotBeta > maxAlphaDotBeta) {
						maxAlphaDotBeta = alphaDotBeta;
						maxAlphaIndex = alphaIndex;
					}
				}

				// Now we can compute the alpha-vector component for this observation, since we have the max.
				// We will need to compute the dot product anyway, so let's just distribute the belief over the
				// sum over observations, and add it all up here.
				for (unsigned int s = 0; s < n; s++) {
					// We compute the value of this state in the alpha-vector, then multiply it by the belief, and add it to
					// the current dot product value for this alpha-vector.
					float value = 0.0f;
					for (unsigned int sp = 0; sp < n; sp++) {
						value += T[s * m * n + action * n + sp] * O[action * n * z + sp * z + observation] * Gamma[maxAlphaIndex * n + sp];
					}
					alphaBA[beliefIndex * n + s] += gamma * value;
				}
			}

			// Once the potential alpha-vector has been computed, compute the value with respect to the belief state.
			float actionValue = 0.0f;
			for (unsigned int s = 0; s < n; s++) {
				actionValue += alphaBA[beliefIndex * n + s] * B[beliefIndex * n + s];
			}

			// If this was larger, then overwrite piPrime and GammaPrime's values.
			if (actionValue > maxActionValue) {
				maxActionValue = actionValue;

				piPrime[beliefIndex] = action;
				for (unsigned int s = 0; s < n; s++) {
					GammaPrime[beliefIndex * n + s] = alphaBA[beliefIndex * n + s];
				}
			}
		}
	}
}

__global__ void lpbvi_restrict_actions(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
		const float *B, const float *T, const float *O, const float *R, float eta,
		const float *Gamma, const unsigned int *pi,
		bool *A)
{
	// Each block will run a different belief. Our overall goal: Restrict the actions
	// within A[beliefIndex * n + action] for all actions a.
	unsigned int beliefIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (beliefIndex >= r) {
		return;
	}

	// First, compute the optimal value at this belief point.
	float maxAlphaDotBeta = 0.0f;

	for (unsigned int alphaIndex = 0; alphaIndex < r; alphaIndex++) {
		float alphaDotBeta = 0.0f;

		for (unsigned int s = 0; s < n; s++) {
			alphaDotBeta += Gamma[alphaIndex * n + s] * B[beliefIndex * n + s];
		}

		// Store the maximal value and index.
		if (alphaIndex == 0 || alphaDotBeta > maxAlphaDotBeta) {
			maxAlphaDotBeta = alphaDotBeta;
		}
	}

	// Assign all actions as not available.
	for (unsigned int action = 0; action < m; action++) {
		A[beliefIndex * n + action] = false;
	}

	// Now that we have the optimal value at this belief point, we can run over the
	// vectors again, and if the value at that belief state is within eta, then
	// we can mark the action in A as allowable.
	for (unsigned int alphaIndex = 0; alphaIndex < r; alphaIndex++) {
		float alphaDotBeta = 0.0f;

		for (unsigned int s = 0; s < n; s++) {
			alphaDotBeta += Gamma[alphaIndex * n + s] * B[beliefIndex * n + s];
		}

		if (maxAlphaDotBeta - alphaDotBeta < eta) {
			A[beliefIndex * n + pi[alphaIndex]] = true;
		}
	}
}

int lpbvi_cuda(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
		bool *A, const float *d_B,
		const float *d_T, const float *d_O, const float *d_R,
		float gamma, float eta, unsigned int horizon,
		unsigned int numThreads,
		float *Gamma, unsigned int *pi)
{
	// The device pointers for the alpha-vectors: Gamma and GammaPrime.
	float *d_Gamma;
	float *d_GammaPrime;

	// The device pointers for the actions taken on each alpha-vector: pi and piPrime.
	unsigned int *d_pi;
	unsigned int *d_piPrime;

	// The device pointer for the intermediate alpha-vectors computed in the inner for loop.
	float *d_AlphaBA;

	// Ensure the data is valid.
	if (n == 0 || m == 0 || z == 0 || r == 0 ||
			A == nullptr || d_B == nullptr ||
			d_T == nullptr || d_O == nullptr || d_R == nullptr ||
			gamma < 0.0 || gamma >= 1.0 || horizon < 1) {
		fprintf(stderr, "Error[lpbvi_cuda]: %s", "Invalid arguments.");
		return -1;
	}

	// Ensure threads are correct.
	if (numThreads % 32 != 0) {
		fprintf(stderr, "Error[lpbvi_cuda]: %s", "Invalid number of threads.");
		return -2;
	}

	unsigned int numBlocks = (unsigned int)((float)r / (float)numThreads) + 1;

	// Allocate the memory on the device for A, and copy the current values.
	bool *d_A;
	if (cudaMalloc(&d_A, r * m * sizeof(bool)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_cuda]: %s",
				"Failed to allocate device-side memory for the actions.");
		return -3;
	}
	if (cudaMemcpy(d_A, A, r * m * sizeof(bool), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_cuda]: %s",
				"Failed to copy memory from host to device for the actions.");
		return -3;
	}

	// Create the device-side Gamma.
	if (cudaMalloc(&d_Gamma, r * n * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_cuda]: %s",
				"Failed to allocate device-side memory for Gamma.");
		return -3;
	}
	if (cudaMemcpy(d_Gamma, Gamma, r * n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_cuda]: %s",
				"Failed to copy memory from host to device for Gamma.");
		return -3;
	}

	if (cudaMalloc(&d_GammaPrime, r * n * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_cuda]: %s",
				"Failed to allocate device-side memory for Gamma (prime).");
		return -3;
	}
	if (cudaMemcpy(d_GammaPrime, Gamma, r * n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_cuda]: %s",
				"Failed to copy memory from host to device for Gamma (prime).");
		return -3;
	}

	// Create the device-side pi.
	if (cudaMalloc(&d_pi, r * sizeof(unsigned int)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_cuda]: %s",
				"Failed to allocate device-side memory for pi.");
		return -3;
	}
	if (cudaMalloc(&d_piPrime, r * sizeof(unsigned int)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_cuda]: %s",
				"Failed to allocate device-side memory for pi (prime).");
		return -3;
	}

	// Create the device-side memory for the intermediate variable alphaBA.
	if (cudaMalloc(&d_AlphaBA, r * n * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_cuda]: %s",
				"Failed to allocate device-side memory for alphaBA.");
		return -3;
	}

	// For each of the updates, run PBVI.
	for (int t = 0; t < horizon; t++) {
		fprintf(stdout, "Iteration %i of %i\n", t+1, horizon);

		// Execute a kernel for the first three stages of for-loops: B, A, Z, as a 3d-block,
		// and the 4th stage for-loop over Gamma as the threads.
		if (t % 2 == 0) {
			lpbvi_update<<< numBlocks, numThreads >>>(n, m, z, r,
					d_A, d_B, d_T, d_O, d_R, gamma,
					d_Gamma, d_pi,
					d_AlphaBA,
					d_GammaPrime, d_piPrime);
		} else {
			lpbvi_update<<< numBlocks, numThreads >>>(n, m, z, r,
					d_A, d_B, d_T, d_O, d_R, gamma,
					d_GammaPrime, d_piPrime,
					d_AlphaBA,
					d_Gamma, d_pi);
		}

		// Check if there was an error executing the kernel.
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Error[lpbvi_cuda]: %s",
							"Failed to execute the 'iteration' kernel.");
			return -3;
		}

		// Wait for the kernel to finish before looping more.
		if (cudaDeviceSynchronize() != cudaSuccess) {
			fprintf(stderr, "Error[lpbvi_cuda]: %s",
							"Failed to synchronize the device.");
			return -3;
		}
	}
	// Copy the final result of Gamma and pi to the variables. This assumes
	// that the memory has been allocated.
	if (horizon % 2 == 1) {
		if (cudaMemcpy(Gamma, d_Gamma, r * n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
			fprintf(stderr, "Error[lpbvi_cuda]: %s",
					"Failed to copy memory from device to host for Gamma.");
			return -3;
		}
		if (cudaMemcpy(pi, d_pi, r * sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess) {
			fprintf(stderr, "Error[lpbvi_cuda]: %s",
					"Failed to copy memory from device to host for pi.");
			return -3;
		}
	} else {
		if (cudaMemcpy(Gamma, d_GammaPrime, r * n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
			fprintf(stderr, "Error[lpbvi_cuda]: %s",
					"Failed to copy memory from device to host for Gamma (prime).");
			return -3;
		}
		if (cudaMemcpy(pi, d_piPrime, r * sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess) {
			fprintf(stderr, "Error[lpbvi_cuda]: %s",
					"Failed to copy memory from device to host for pi (prime).");
			return -3;
		}
	}

	// Once freed, compute the available actions for the next iteration.
	lpbvi_restrict_actions<<< numBlocks, numThreads >>>(n, m, z, r,
					d_B, d_T, d_O, d_R, eta,
					d_Gamma, d_pi, d_A);

	// Check if there was an error executing the kernel.
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_cuda]: %s",
						"Failed to execute the 'iteration' kernel.");
		return -3;
	}

	// Wait for the kernel to finish.
	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_cuda]: %s",
						"Failed to synchronize the device.");
		return -3;
	}

	// Copy the result to the r-n array A.
	if (cudaMemcpy(A, d_A, r * m * sizeof(bool), cudaMemcpyDeviceToHost) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_cuda]: %s",
				"Failed to copy memory from device to host for the available actions at each belief state A.");
		return -3;
	}

	// Free the device-side Gamma and pi.
	cudaFree(d_A);
	cudaFree(d_Gamma);
	cudaFree(d_GammaPrime);
	cudaFree(d_pi);
	cudaFree(d_piPrime);
	cudaFree(d_AlphaBA);

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

int lpbvi_uninitialize(float *&d_B, float *&d_T, float *&d_O, float **&d_R, unsigned int k)
{
	if (d_B != nullptr) {
		cudaFree(d_B);
	}
	d_B = nullptr;

	if (d_T != nullptr) {
		cudaFree(d_T);
	}
	d_T = nullptr;

	if (d_O != nullptr) {
		cudaFree(d_O);
	}
	d_O = nullptr;

	if (d_R != nullptr) {
		for (unsigned int i = 0; i < k; i++) {
			cudaFree(d_R[i]);
		}
	}

	return 0;
}
