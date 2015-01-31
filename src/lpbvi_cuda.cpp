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

#include <chrono>

LPBVICuda::LPBVICuda() : LPBVI()
{
	d_B = nullptr;
	d_T = nullptr;
	d_O = nullptr;
	d_R = nullptr;
	k = 1;
	d_NonZeroBeliefStates = nullptr;
	d_SuccessorStates = nullptr;
	maxNonZeroBeliefStates = 1;
	maxSuccessorStates = 1;
}

LPBVICuda::~LPBVICuda()
{
	uninitialize_variables();
}

void LPBVICuda::set_performance_variables(unsigned int nonZeroBeliefStates, unsigned int successorStates)
{
	maxNonZeroBeliefStates = nonZeroBeliefStates;
	maxSuccessorStates = successorStates;
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

	// After setting up everything, begin timing.
	auto start = std::chrono::high_resolution_clock::now();

	std::cout << "Starting...\n"; std::cout.flush();

	// For each reward function, execute the CUDA code which computes the value and limits the actions for the next level.
	for (unsigned int i = 0; i < R->get_num_rewards(); i++) {
		std::cout << "  R[" << i << "]" << std::endl; std::cout.flush();

		SARewards *Ri = dynamic_cast<SARewards *>(R->get(i));

		// Define eta (the one-step slack).
		double etai = delta[i];
		if (constrainEta) {
			double deltaB = compute_belief_density(S);
			double epsiloni = (Ri->get_max() - Ri->get_min()) / (1.0 - h->get_discount_factor()) * deltaB;
			etai = std::max(0.0, (1.0 - h->get_discount_factor()) * delta[i] - epsiloni);
		}

		// Create Gamma and pi.
		float *Gamma = new float[B.size() * S->get_num_states()];
		for (unsigned int x = 0; x < B.size(); x++) {
			for (unsigned int y = 0; y < S->get_num_states(); y++) {
//				Gamma[x * S->get_num_states() + y] = Ri->get_min() / (1.0 - h->get_discount_factor());
				Gamma[x * S->get_num_states() + y] = 0.0f;
			}
		}
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
				d_NonZeroBeliefStates,
				maxNonZeroBeliefStates,
				d_SuccessorStates,
				maxSuccessorStates,
				h->get_discount_factor(),
				etai,
				updates,
				1024, // Number of Threads
				Gamma,
				pi);

//		// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//		// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//		// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//		// Print out Gamma and pi.
//		if (i == 0) {
//			std::cout << "Gamma:" << std::endl;
//			for (unsigned int x = 0; x < B.size(); x++) {
//				for (unsigned int y = 0; y < S->get_num_states(); y++) {
//					std::cout << y << ":" << Gamma[x * S->get_num_states() + y] << " ";
//				}
//				std::cout << std::endl;
//			}
//			std::cout.flush();
//
//			std::cout << "pi:" << std::endl;
//			for (unsigned int x = 0; x < B.size(); x++) {
//				std::cout << x << ":" << pi[x] << " ";
//			}
//			std::cout.flush();
//		}
//		// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//		// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//		// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG

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

	std::cout << "Complete LPBVI." << std::endl; std::cout.flush();

	// After the main loop is complete, end timing. Also, output the result of the computation time.
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Total Elapsed Time (GPU Version): " << elapsed.count() << std::endl; std::cout.flush();

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
	std::cout << "Creating B... "; std::cout.flush();

	float *Barray = new float[B.size() * S->get_num_states()];
	unsigned int counter = 0;
	for (BeliefState *b : B) {
		for (unsigned int i = 0; i < S->get_num_states(); i++) {
			Barray[counter * S->get_num_states() + i] = b->get(S->get(i));
		}
		counter++;
	}

//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//	for (unsigned int i = 0; i < B.size(); i++) {
//		for (unsigned int j = 0; j < S->get_num_states(); j++) {
//			std::cout << Barray[i * S->get_num_states() + j] << " ";
//		}
//		std::cout << std::endl;
//	}
//	std::cout.flush();
//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG

	std::cout << "Transferring B... "; std::cout.flush();

	int result = lpbvi_initialize_belief_points(S->get_num_states(), B.size(), Barray, d_B);
	delete [] Barray;
	if (result != 0) {
		throw PolicyException();
	}

	std::cout << "Done.\nTransferring T... "; std::cout.flush();

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

	std::cout << "Done.\nTransferring O... "; std::cout.flush();

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

	std::cout << "Done.\nTransferring R... "; std::cout.flush();

	k = R->get_num_rewards();
	d_R = new float*[k];
	for (unsigned int i = 0; i < k; i++) {
		SARewardsArray *Ri = dynamic_cast<SARewardsArray *>(R->get(i));
		if (Ri == nullptr) {
			throw PolicyException();
		}

		std::cout << (i + 1) << " "; std::cout.flush();

		result = lpbvi_initialize_rewards(S->get_num_states(),
				A->get_num_actions(),
				Ri->get_rewards(),
				d_R[i]);
		if (result != 0) {
			throw PolicyException();
		}
	}

	std::cout << ". Done.\nCreating Non-Zero Belief States... "; std::cout.flush();

	// Purposefully an int for having the sign bit store the termination point in the array's row.
	// This stores the hash values of the states (which in our case will be indexes).
	int *nonZeroBeliefStates = new int[B.size() * maxNonZeroBeliefStates];
	counter = 0;

	for (BeliefState *b : B) {
		unsigned int counterOverStates = 0;

		for (auto state : *S) {
			State *s = resolve(state);

			if (b->get(s) > 0.0) {
				// Note: 'counter' works here because B is a vector, which is ordered, so the for loop
				// steps over the belief set in order anyway.
				nonZeroBeliefStates[counter * maxNonZeroBeliefStates + counterOverStates] = s->hash_value();
				counterOverStates++;
			}

			if (counterOverStates == maxNonZeroBeliefStates) {
				break;
			}
		}

		if (counterOverStates < maxNonZeroBeliefStates) {
			nonZeroBeliefStates[counter * maxNonZeroBeliefStates + counterOverStates] = -1;
		}

		counter++;
	}

//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//	for (unsigned int i = 0; i < maxNonZeroBeliefStates; i++) {
//		for (unsigned int j = 0; j < B.size(); j++) {
//			std::cout << nonZeroBeliefStates[j * maxNonZeroBeliefStates + i] << "\t";
//		}
//		std::cout << std::endl;
//	}
//	std::cout.flush();
//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG

	std::cout << "Transferring Non-Zero Belief States... "; std::cout.flush();

	result = lpbvi_initialize_nonzero_beliefs(B.size(), maxNonZeroBeliefStates,
			nonZeroBeliefStates, d_NonZeroBeliefStates);
	delete [] nonZeroBeliefStates;
	if (result != 0) {
		throw PolicyException();
	}

	std::cout << "Done.\nCreating Successor States... "; std::cout.flush();

	// Similarly, this holds the successor state hash values (which in our case are indexes) for
	// each state-action pair.
	int *successorStates = new int[S->get_num_states() * A->get_num_actions() * maxSuccessorStates];
	counter = 0;

	for (auto state : *S) {
		State *s = resolve(state);
		unsigned int counterOverActions = 0;

		for (auto action : *A) {
			Action *a = resolve(action);
			unsigned int counterOverNextStates = 0;

			for (auto statePrime : *S) {
				State *sp = resolve(statePrime);

				if (T->get(s, a, sp) > 0.0) {
					successorStates[s->hash_value() * A->get_num_actions() * maxSuccessorStates +
									a->hash_value() * maxSuccessorStates +
									counterOverNextStates] = sp->hash_value();
					counterOverNextStates++;
				}

				if (counterOverNextStates == maxSuccessorStates) {
					break;
				}
			}

			if (counterOverNextStates < maxSuccessorStates) {
				successorStates[s->hash_value() * A->get_num_actions() * maxSuccessorStates +
							a->hash_value() * maxSuccessorStates +
							counterOverNextStates] = -1;
			}

			counterOverActions++;
		}

		counter++;
	}

//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//	for (unsigned int i = 0; i < S->get_num_states(); i++) {
//		std::cout << "S = " << i << ":\n";
//		for (unsigned int j = 0; j < A->get_num_actions(); j++) {
//			std::cout << "<" << successorStates[i * A->get_num_actions() * maxSuccessorStates +
//										 j * maxSuccessorStates + 0]
//						<< ", "
//						<< successorStates[i * A->get_num_actions() * maxSuccessorStates +
//										 j * maxSuccessorStates + 1] << "> ";
//		}
//		std::cout << std::endl;
//	}
//	std::cout.flush();
//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
//	// DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG

	std::cout << "Transferring Successor States... "; std::cout.flush();

	result = lpbvi_initialize_successors(S->get_num_states(), A->get_num_actions(), maxSuccessorStates,
			successorStates, d_SuccessorStates);
	delete [] successorStates;
	if (result != 0) {
		throw PolicyException();
	}

	std::cout << "Done.\nCompleted Variable Initialization " << std::endl; std::cout.flush();
}

void LPBVICuda::uninitialize_variables()
{
	lpbvi_uninitialize(d_B, d_T, d_O, d_R, k, d_NonZeroBeliefStates, d_SuccessorStates);

	delete [] d_R;
	d_R = nullptr;
}
