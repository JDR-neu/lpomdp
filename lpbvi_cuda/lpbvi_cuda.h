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


// Note: The "EXT" part ensures this name is different from the define in the header
// file inside the C++ code.
#ifndef LPBVI_CUDA_EXT_H
#define LPBVI_CUDA_EXT_H


/**
 * Execute PBVI for the infinite horizon POMDP model specified, except this time we
 * limit actions taken at a state to within an array of available actions. Also, adjust A
 * after completion to be within slack for each belief point.
 * @param	n						The number of states.
 * @param	m						The number of actions, in total, that are possible.
 * @param	z						The number of observations.
 * @param	r						The number of belief points.
 * @param	A						A mapping of belief-action pairs (r-m array) to a boolean if the action
 * 									is available at that belief state or not. This will be modified.
 * @param	d_B						A r-n array, consisting of r sets of n-vector belief distributions.
 *  								(Device-side pointer.)
 * @param	d_T						A mapping of state-action-state triples (n-m-n array) to a
 * 									transition probability. (Device-side pointer.)
 * @param	d_O						A mapping of action-state-observations triples (m-n-z array) to a
 * 									transition probability. (Device-side pointer.)
 * @param	d_R						A mapping of state-action triples (n-m array) to a reward.
 * 									(Device-side pointer.)
 * @param	d_NonZeroBeliefStates	A mapping from beliefs to an array of state indexes.
 * 									(Device-side pointer.)
 * @param	maxNonZeroBeliefStates	The maximum number of non-zero belief states.
 * @param	d_SuccessorStates		A mapping from state-action pairs to successor state indexes.
 * 									(Device-side pointer.)
 * @param	maxSuccessorStates		The maximum number of successor states possible.
 * @param	gamma					The discount factor in [0.0, 1.0).
 * @param	eta						The tolerable deviation from optimal.
 * @param	horizon					How many time steps to iterate.
 * @param	numThreads				The number of CUDA threads per block. Use 128, 256, 512, or 1024 (multiples of 32).
 * @param	Gamma					The resultant policy; set of alpha vectors (r-n array). This will be modified.
 * @param	pi						The resultant policy; one action for each alpha-vector (r-array).
 * 									This will be modified.
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -2 if the number
 * 			of blocks and threads is less than the number of states; -3 if an error with
 * 			the CUDA functions arose.
 */
int lpbvi_cuda(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
		bool *A, const float *d_B,
		const float *d_T, const float *d_O, const float *d_R,
		const int *d_NonZeroBeliefStates, unsigned int maxNonZeroBeliefStates,
		const int *d_SuccessorStates, unsigned int maxSuccessorStates,
		float gamma, float eta, unsigned int horizon,
		unsigned int numThreads,
		float *Gamma, unsigned int *pi);

/**
 * Initialize CUDA by transferring all of the constant LPOMDP model information to the device.
 * @param	n			The number of states.
 * @param	r			The number of belief points.
 * @param	B			A r-n array, consisting of r sets of n-vector belief distributions.
 * @param	d_B			A r-n array, consisting of r sets of n-vector belief distributions.
 *  					(Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int lpbvi_initialize_belief_points(unsigned int n, unsigned int r, const float *B, float *&d_B);

/**
 * Initialize CUDA by transferring all of the constant LPOMDP model information to the device.
 * @param	n			The number of states.
 * @param	m			The number of actions, in total, that are possible.
 * @param	T			A mapping of state-action-state triples (n-m-n array) to a
 * 						transition probability.
 * @param	d_T			A mapping of state-action-state triples (n-m-n array) to a
 * 						transition probability. (Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int lpbvi_initialize_state_transitions(unsigned int n, unsigned int m, const float *T, float *&d_T);

/**
 * Initialize CUDA by transferring all of the constant LPOMDP model information to the device.
 * @param	n			The number of states.
 * @param	m			The number of actions, in total, that are possible.
 * @param	z			The number of observations.
 * @param	O			A mapping of action-state-observation triples (m-n-z array) to a
 * 						transition probability.
 * @param	d_O			A mapping of action-state-observation triples (m-n-z array) to a
 * 						transition probability. (Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int lpbvi_initialize_observation_transitions(unsigned int n, unsigned int m, unsigned int z, const float *O, float *&d_O);

/**
 * Initialize CUDA by transferring all of the constant LPOMDP model information to the device.
 * @param	n			The number of states.
 * @param	m			The number of actions, in total, that are possible.
 * @param	R			A mapping of state-action pairs (n-m array) to a reward.
 * @param	d_R			A mapping of state-action pairs (n-m array) to a reward.
 * 						(Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int lpbvi_initialize_rewards(unsigned int n, unsigned int m, const float *R, float *&d_R);

/**
 * Initialize CUDA by transferring all of the constant LPOMDP model information to the device.
 * @param	r						The number of belief points.
 * @param	maxNonZeroBeliefStates	The maximum number of non-zero belief states.
 * @param	nonZeroBeliefStates		A mapping of beliefs to an array of state indexes;
 * 									-1 denotes the end of the array.
 * @param	d_NonZeroBeliefStates	A mapping of beliefs to an array of state indexes;
 * 									-1 denotes the end of the array. (Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int lpbvi_initialize_nonzero_beliefs(unsigned int r, unsigned int maxNonZeroBeliefStates,
		int *nonZeroBeliefStates, int *&d_NonZeroBeliefStates);

/**
 * Initialize CUDA by transferring all of the constant LPOMDP model information to the device.
 * @param	n					The number of states.
 * @param	m					The number of actions, in total, that are possible.
 * @param	maxSuccessorStates	The maximum number of successor states.
 * @param	successorStates		A mapping of state-action pairs a set of possible successor states;
 * 								-1 denotes the end of the array.
 * @param	d_SuccessorStates	A mapping of state-action pairs a set of possible successor states;
 * 								-1 denotes the end of the array. (Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int lpbvi_initialize_successors(unsigned int n, unsigned int m, unsigned int maxSuccessorStates,
		int *successorStates, int *&d_SuccessorStates);

/**
 * Uninitialize CUDA by freeing all of the constant MDP model information on the device.
 * @param	d_B						A r-n array, consisting of r sets of n-vector belief distributions.
 *  								(Device-side pointer.)
 * @param	d_T						A mapping of state-action-state triples (n-m-n array) to a
 * 									transition probability. (Device-side pointer.)
 * @param	d_O						A mapping of action-state-observation triples (m-n-z array) to a
 * 									transition probability. (Device-side pointer.)
 * @param	d_R						A k-array, each mapping of state-action pairs (n-m array) to a reward.
 * 									(Device-side pointer.)
 * @param	k						The number of rewards.
 * @param	d_NonZeroBeliefStates	A mapping from beliefs to an array of state indexes.
 * 									(Device-side pointer.)
 * @param	d_SuccessorStates		A mapping from state-action pairs to successor state indexes.
 * 									(Device-side pointer.)
 * @return	Returns 0 upon success.
 */
int lpbvi_uninitialize(float *&d_B, float *&d_T, float *&d_O, float **&d_R, unsigned int k,
		int *&d_NonZeroBeliefStates, int *&d_SuccessorStates);


#endif // LPBVI_CUDA_EXT_H
