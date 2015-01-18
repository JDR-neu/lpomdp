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

#include <unistd.h>

#include "../include/lpbvi_cuda.h"
#include "../include/lpomdp.h"

#include "../lpbvi_cuda/lpbvi_cuda.h"

#include "../../librbr/librbr/include/pomdp/pomdp_utilities.h"

#include "../../librbr/librbr/include/management/conversion.h"

#include "../../librbr/librbr/include/core/states/indexed_state.h"
#include "../../librbr/librbr/include/core/actions/indexed_action.h"
#include "../../librbr/librbr/include/core/observations/indexed_observation.h"

#include "../../librbr/librbr/include/core/state_transitions/state_transitions_array.h"
#include "../../librbr/librbr/include/core/observation_transitions/observation_transitions_array.h"
#include "../../librbr/librbr/include/core/rewards/sa_rewards_array.h"

#include "../../librbr/librbr/include/core/core_exception.h"
#include "../../librbr/librbr/include/core/states/state_exception.h"
#include "../../librbr/librbr/include/core/actions/action_exception.h"
#include "../../librbr/librbr/include/core/observations/observation_exception.h"
#include "../../librbr/librbr/include/core/state_transitions/state_transition_exception.h"
#include "../../librbr/librbr/include/core/observation_transitions/observation_transition_exception.h"
#include "../../librbr/librbr/include/core/rewards/reward_exception.h"
#include "../../librbr/librbr/include/core/policy/policy_exception.h"

#include "../../librbr/librbr/include/core/actions/action_utilities.h"

#include <iostream>

#include <math.h>
#include <vector>
#include <unordered_map>
#include <algorithm>

LPBVICuda::LPBVICuda() : LPBVI()
{
	d_B = nullptr;
	d_T = nullptr;
	d_O = nullptr;
	d_R = nullptr;
	k = 1;
}

LPBVICuda::~LPBVICuda()
{
	uninitialize_variables();
}

PolicyAlphaVectors **LPBVICuda::solve_infinite_horizon(StatesMap *S, ActionsMap *A,
		ObservationsMap *Z, StateTransitions *T, ObservationTransitions *O,
		FactoredRewards *R, Horizon *h, std::vector<float> &delta)
{
	// Ensure states, actions, and observations are indexed.
	for (auto s : *S) {
		IndexedState *state = dynamic_cast<IndexedState *>(resolve(s));
		if (state == nullptr) {
			throw PolicyException();
		}
	}
	for (auto a : *A) {
		IndexedAction *action = dynamic_cast<IndexedAction *>(resolve(a));
		if (action == nullptr) {
			throw PolicyException();
		}
	}
	for (auto z : *Z) {
		IndexedObservation *observation = dynamic_cast<IndexedObservation *>(resolve(z));
		if (observation == nullptr) {
			throw PolicyException();
		}
	}

	// The final set of alpha vectors.
	PolicyAlphaVectors **policy = new PolicyAlphaVectors*[R->get_num_rewards()];
	for (int i = 0; i < (int)R->get_num_rewards(); i++) {
		policy[i] = new PolicyAlphaVectors(h->get_horizon());
	}

	// Initialize the set of belief points to be the initial set. This must be a copy, since memory is managed
	// for both objects independently.
	for (BeliefState *b : initialB) {
		B.push_back(new BeliefState(*b));
	}

	std::cout << "Initial Num Belief Points: " << initialB.size() << std::endl; std::cout.flush();

	// Initialize variables for CUDA.
	initialize_variables(S, A, Z, T, O, R, h, delta);

	// Setup the array of actions available for each belief point. They are all available to start.
	bool *available = new bool[B.size() * A->get_num_actions()];
	for (unsigned int i = 0; i < B.size() * A->get_num_actions(); i++) {
		available[i] = true;
	}

	// For each reward function, execute the CUDA code which computes the value and limits the actions for the next level.
	for (unsigned int i = 0; i < R->get_num_rewards(); i++) {
		std::cout << "  R[" << i << "]" << std::endl; std::cout.flush();

		// Define eta (the one-step slack).
		double etai = delta[i];
		if (constrainEta) {
			SARewards *Ri = dynamic_cast<SARewards *>(R->get(i));
			double deltaB = compute_belief_density(S);
			double epsiloni = (Ri->get_max() - Ri->get_min()) / (1.0 - h->get_discount_factor()) * deltaB;
			etai = std::max(0.0, (1.0 - h->get_discount_factor()) * delta[i] - epsiloni);
		}

		// Create Gamma and pi.
		float *Gamma = new float[B.size() * S->get_num_states()];
		unsigned int *pi = new unsigned int[B.size()];

		// Execute CUDA!
		lpbvi_cuda(S->get_num_states(),
				A->get_num_actions(),
				Z->get_num_observations(),
				B.size(),
				available,
				d_B,
				d_T,
				d_O,
				d_R[i],
				h->get_discount_factor(),
				etai,
				updates,
				1024, // Number of Threads
				Gamma,
				pi);

		// Create the vector of policy alpha vectors and set the policy equal to them.
		// Note: This transfer responsibility of memory management to the policy variable.
		std::vector<PolicyAlphaVector *> GammaAlphaVectors;
		for (unsigned int j = 0; j < B.size(); j++) {
			PolicyAlphaVector *alpha = new PolicyAlphaVector(A->get(pi[j]));
			for (auto s : *S) {
				State *state = resolve(s);
				alpha->set(state, Gamma[j * S->get_num_states() + state->hash_value()]);
			}
			GammaAlphaVectors.push_back(alpha);
		}
		policy[i]->set(GammaAlphaVectors);

		// Free the memory which was allocated.
		delete [] Gamma;
		delete [] pi;
	}

	// Uninitialize the CUDA variables.
	uninitialize_variables();

	// Free the available array.
	delete [] available;
	available = nullptr;

	return policy;
}

void LPBVICuda::initialize_variables(StatesMap *S, ActionsMap *A, ObservationsMap *Z,
		StateTransitions *T, ObservationTransitions *O, FactoredRewards *R,
		Horizon *h, std::vector<float> &delta)
{
	float *Barray = new float[B.size() * S->get_num_states()];
	unsigned int counter = 0;
	for (BeliefState *b : B) {
		for (unsigned int i = 0; i < S->get_num_states(); i++) {
			Barray[counter * S->get_num_states() + i] = b->get(S->get(i));
		}
		counter++;
	}

	int result = lpbvi_initialize_belief_points(S->get_num_states(), B.size(), Barray, d_B);
	delete [] Barray;
	if (result != 0) {
		throw PolicyException();
	}

	StateTransitionsArray *Tarray = dynamic_cast<StateTransitionsArray *>(T);
	if (Tarray == nullptr) {
		throw PolicyException();
	}

	result = lpbvi_initialize_state_transitions(S->get_num_states(),
			A->get_num_actions(),
			Tarray->get_state_transitions(),
			d_T);
	if (result != 0) {
		throw PolicyException();
	}

	ObservationTransitionsArray *Oarray = dynamic_cast<ObservationTransitionsArray *>(O);
	if (Oarray == nullptr) {
		throw PolicyException();
	}

	result = lpbvi_initialize_observation_transitions(S->get_num_states(),
			A->get_num_actions(),
			Z->get_num_observations(),
			Oarray->get_observation_transitions(),
			d_O);
	if (result != 0) {
		throw PolicyException();
	}

	k = R->get_num_rewards();
	d_R = new float*[k];
	for (unsigned int i = 0; i < k; i++) {
		SARewardsArray *Ri = dynamic_cast<SARewardsArray *>(R->get(i));

		result = lpbvi_initialize_rewards(S->get_num_states(),
				A->get_num_actions(),
				Ri->get_rewards(),
				d_R[i]);
		if (result != 0) {
			throw PolicyException();
		}
	}
}

void LPBVICuda::uninitialize_variables()
{
	lpbvi_uninitialize(d_B, d_T, d_O, d_R, k);

	delete [] d_R;
	d_R = nullptr;
}
