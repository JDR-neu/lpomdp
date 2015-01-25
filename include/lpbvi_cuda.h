/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2015 Kyle Wray, University of Massachusetts
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


#include "lpomdp.h"
#include "lpbvi.h"

/**
 * Solve a Lexicographic Partially Observable Markov Decision Process (LMDP) using CUDA.
 */
class LPBVICuda : public LPBVI {
public:
	/**
	 * The default constructor for the LPBVICuda class. Default number of iterations for infinite
	 * horizon POMDPs is 1. The default expansion rule is Random Belief Selection.
	 */
	LPBVICuda();

	/**
	 * A constructor for the LPBVICuda class which allows for the specification of the expansion rule,
	 * and the number of iterations (both updates and expansions) to run for infinite horizon.
	 * The default is 1 for both.
	 * @param	expansionRule			The expansion rule to use.
	 * @param	updateIterations 		The number of update iterations to run for infinite horizon POMDPs.
	 * @param	expansionIterations 	The number of expansion iterations to run for infinite horizon POMDPs.
	 */
	LPBVICuda(POMDPPBVIExpansionRule expansionRule, unsigned int updateIterations, unsigned int expansionIterations);

	/**
	 * The deconstructor for the LPBVICuda class. This method frees all the belief state memory.
	 */
	virtual ~LPBVICuda();

	/**
	 * Assign the max non-zero belief point states and successor states.
	 * @param	nonZeroBeliefStates		The max non-zero belief states.
	 * @param	sucessorStates			The max successor states.
	 */
	void set_performance_variables(unsigned int nonZeroBeliefStates, unsigned int successorStates);

protected:
	/**
	 * Solve an infinite horizon LMDP using value iteration.
	 * @param	S					The finite states.
	 * @param	A					The finite actions.
	 * @param	Z					The finite observations.
	 * @param	T					The finite state transition function.
	 * @param	O					The finite observation transition function.
	 * @param	R					The factored state-action rewards.
	 * @param	h					The horizon.
	 * @param	delta				The slack vector.
	 * @throw	PolicyException		An error occurred computing the policy.
	 * @return	Return the optimal policy.
	 */
	virtual PolicyAlphaVectors **solve_infinite_horizon(StatesMap *S, ActionsMap *A,
			ObservationsMap *Z, StateTransitions *T, ObservationTransitions *O,
			FactoredRewards *R, Horizon *h, std::vector<float> &delta);

	/**
	 * Initialize the variables by creating the device-side memory.
	 * @param	S					The finite states.
	 * @param	A					The finite actions.
	 * @param	Z					The finite observations.
	 * @param	T					The finite state transition function.
	 * @param	O					The finite observation transition function.
	 * @param	R					The factored state-action rewards.
	 * @param	h					The horizon.
	 * @param	delta				The slack vector.
	 */
	virtual void initialize_variables(StatesMap *S, ActionsMap *A, ObservationsMap *Z,
			StateTransitions *T, ObservationTransitions *O, FactoredRewards *R,
			Horizon *h, std::vector<float> &delta);

	/**
	 * Uninitialize the variables for the device-side memory.
	 */
	virtual void uninitialize_variables();

	/**
	 * A quick helper array of actions arranged by their hash value.
	 */
	std::vector<Action *> sortedActions;

	/**
	 * The device-side pointer to the memory location of belief points.
	 */
	float *d_B;

	/**
	 * The device-side pointer to the memory location of state transitions.
	 */
	float *d_T;

	/**
	 * The device-side pointer to the memory location of observation transitions.
	 */
	float *d_O;

	/**
	 * The device-side pointer to the memory location of rewards, one for each reward.
	 */
	float **d_R;

	/**
	 * The number of rewards.
	 */
	unsigned int k;

	/**
	 * The device-side pointer to the memory location of the non-zero belief states.
	 */
	int *d_NonZeroBeliefStates;

	/**
	 * The device-side pointer to the memory location of the successor states.
	 */
	int *d_SuccessorStates;

	/**
	 * The maximum number of states with non-zero belief probabilities that is possible.
	 */
	unsigned int maxNonZeroBeliefStates;

	/**
	 * The maximum number of successor states given any state-action pair.
	 */
	unsigned int maxSuccessorStates;

};


#endif // LPBVI_CUDA_H
