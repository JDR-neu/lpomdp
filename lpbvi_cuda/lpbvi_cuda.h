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


#ifndef LPBVI_CUDA_H
#define LPBVI_CUDA_H


/**
 * Execute value iteration for the infinite horizon MDP model specified, except this time we
 * limit actions taken at a state to within an array of available actions.
 * @param	n			The number of states.
 * @param	m			The number of actions, in total, that are possible.
 * @param	z			The number of observations.
 * @param	r			The number of belief points.
 * @param	d_A			A mapping of states-action pairs (n-m array) to a boolean if the action
 * 						is available at that state or not. (Device-side pointer.)
 * @param	d_B			A r-n array, consisting of r sets of n-vector belief distributions.
 *  					(Device-side pointer.)
 * @param	d_T			A mapping of state-action-state triples (n-m-n array) to a
 * 						transition probability. (Device-side pointer.)
 * @param	d_O			A mapping of action-state-observations triples (m-n-z array) to a
 * 						transition probability. (Device-side pointer.)
 * @param	d_R			A mapping of state-action triples (n-m array) to a reward.
 * 						(Device-side pointer.)
 * @param	Rmin		The minimum reward possible, for use in computing the number
 * 						of iterations.
 * @param	Rmax		The maximum reward possible, for use in computing the number
 * 						of iterations.
 * @param	gamma		The discount factor in [0.0, 1.0).
 * @param	horizon		How many time steps to iterate.
 * @param	numBlocks	The number of CUDA blocks. Ensure that numBlocks * numThreads >= n.
 * @param	numThreads	The number of CUDA threads per block. Use 128, 256, 512, or 1024 (multiples of 32).
 * @param	Gamma		The resultant policy; set of alpha vectors. This will be modified.
 * @param	pi			The resultant policy; one action for each alpha-vector (an n-array).
 * 						This will be modified.
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -2 if the number
 * 			of blocks and threads is less than the number of states; -3 if an error with
 * 			the CUDA functions arose.
 */
int lpbvi_cuda(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
		const bool *d_A, const float *d_B,
		const float *d_T, const float *d_O, const float *d_R,
		float Rmin, float Rmax, float gamma, unsigned int horizon,
		unsigned int numBlocks, unsigned int numThreads,
		float *Gamma, unsigned int *pi);

/**
 * Initialize CUDA by transferring all of the constant LPOMDP model information to the device.
 * @param	m			The number of actions, in total, that are possible.
 * @param	r			The number of belief points.
 * @param	A			A belief-action matrix (r-m array) of booleans, which state
 * 						if an action is available at the corresponding belief point.
 * @param	d_A			A belief-action matrix (r-m array) of booleans, which state
 * 						if an action is available at the corresponding belief point.
 * 						(Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int lpbvi_initialize_actions(unsigned int m, unsigned int r, const bool *A, bool *&d_A);

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
 * @param	n				The number of states.
 * @param	m				The number of actions, in total, that are possible.
 * @param	GammaAStar		A vector of size n for each action (m-n array).
 * @param	d_GammaAStar	A vector of size n for each action (m-n array).
 * 							(Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int lpbvi_initialize_gamma_a_star(unsigned int n, unsigned int m, const float *GammaAStar, float *&d_GammaAStar);

/**
 * Uninitialize CUDA by freeing all of the constant MDP model information on the device.
 * @param	d_A				A belief-action matrix (r-m array) of booleans, which state
 * 							if an action is available at the corresponding belief point.
 * 							(Device-side pointer.)
 * @param	d_B				A r-n array, consisting of r sets of n-vector belief distributions.
 *  						(Device-side pointer.)
 * @param	d_T				A mapping of state-action-state triples (n-m-n array) to a
 * 							transition probability. (Device-side pointer.)
 * @param	d_O				A mapping of action-state-observation triples (m-n-z array) to a
 * 							transition probability. (Device-side pointer.)
 * @param	d_R				A mapping of state-action pairs (n-m array) to a reward.
 * 							(Device-side pointer.)
 * @param	d_GammaAStar	A vector of size n for each action (m-n array).
 * 							(Device-side pointer.)
 * @return	Returns 0 upon success.
 */
int lpbvi_uninitialize(bool *&d_A, float *&d_B, float *&d_T, float *&d_O, float *&d_R, float *&d_GammaAStar);


#endif // LPBVI_CUDA_H
