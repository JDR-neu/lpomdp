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

#include "../include/lpbvi.h"

#include "../../librbr/librbr/include/pomdp/pomdp_utilities.h"

#include "../../librbr/librbr/include/management/conversion.h"

#include "../../librbr/librbr/include/core/state_transitions/state_transitions_array.h"
#include "../../librbr/librbr/include/core/rewards/sas_rewards_array.h"

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

LPBVI::LPBVI() : POMDPPBVI()
{
	beliefToRecord = nullptr;
	constrainEta = false;
}

LPBVI::LPBVI(POMDPPBVIExpansionRule expansionRule, unsigned int updateIterations,
		unsigned int expansionIterations) : POMDPPBVI(expansionRule, updateIterations, expansionIterations)
{
	beliefToRecord = nullptr;
	constrainEta = false;
}

LPBVI::~LPBVI()
{
	reset();
}

void LPBVI::compute_num_update_iterations(POMDP *pomdp, double epsilon)
{
	// This must be an LPOMDP.
	LPOMDP *lpomdp = dynamic_cast<LPOMDP *>(pomdp);
	if (lpomdp == nullptr) {
		throw CoreException();
	}

	Horizon *h = lpomdp->get_horizon();
	if (h == nullptr) {
		throw CoreException();
	}

	FactoredRewards *R = lpomdp->get_rewards();
	if (R != nullptr)

	updates = 0;

	for (unsigned int i = 0; i < R->get_num_rewards(); i++) {
		// Attempt to convert the rewards object into SARewards.
		SARewards *Ri = dynamic_cast<SARewards *>(R->get(i));
		if (Ri == nullptr) {
			throw RewardException();
		}

		// Make sure we do not take the log of 0.
		double Rmin = Ri->get_min();
		double Rmax = Ri->get_max();
		if (Rmax - Rmin < 0.000001) {
			Rmax = Rmin + 0.000001;
		}

		updates = (unsigned int)std::max((double)updates, (double)((log(epsilon) - log(Rmax - Rmin)) / log(h->get_discount_factor())));
	}
}

void LPBVI::set_belief_to_record(BeliefState *b)
{
	reset();
	beliefToRecord = new BeliefState(*b);
}

const std::vector<std::vector<double> > &LPBVI::get_recorded_values() const
{
	return recordedValues;
}

void LPBVI::constraint_eta(bool value)
{
	constrainEta = value;
}

PolicyAlphaVectors *LPBVI::solve(POMDP *pomdp)
{
	throw CoreException();
}

PolicyAlphaVectors **LPBVI::solve(LPOMDP *lpomdp)
{
	// Handle the trivial case.
	if (lpomdp == nullptr) {
		return nullptr;
	}

	// Attempt to convert the states object into FiniteStates.
	StatesMap *S = dynamic_cast<StatesMap *>(lpomdp->get_states());
	if (S == nullptr) {
		throw StateException();
	}

	// Attempt to convert the actions object into FiniteActions.
	ActionsMap *A = dynamic_cast<ActionsMap *>(lpomdp->get_actions());
	if (A == nullptr) {
		throw ActionException();
	}

	// Attempt to convert the observations object into FiniteObservations.
	ObservationsMap *Z = dynamic_cast<ObservationsMap *>(lpomdp->get_observations());
	if (Z == nullptr) {
		throw ObservationException();
	}

	// Attempt to get the state transitions.
	StateTransitions *T = lpomdp->get_state_transitions();
	if (T == nullptr) {
		throw StateTransitionException();
	}

	// Attempt to get the observations transitions.
	ObservationTransitions *O = lpomdp->get_observation_transitions();
	if (O == nullptr) {
		throw ObservationTransitionException();
	}

	// Attempt to convert the rewards object into FactoredRewards. Also, ensure that the
	// type of each element is SASRewards.
	FactoredRewards *R = dynamic_cast<FactoredRewards *>(lpomdp->get_rewards());
	if (R == nullptr) {
		throw RewardException();
	}
	/*
	for (int i = 0; i < R->get_num_rewards(); i++) {
		SARewards *Ri = dynamic_cast<SARewards *>(R->get(i));
		if (Ri == nullptr) {
			throw RewardException();
		}
	}
	*/

	// Handle the other trivial case in which the slack variables were incorrectly defined.
	if (lpomdp->get_slack().size() != R->get_num_rewards()) {
		throw RewardException();
	}
	for (int i = 0; i < (int)lpomdp->get_slack().size(); i++) {
		if (lpomdp->get_slack().at(i) < 0.0) {
			throw RewardException();
		}
	}

	// Obtain the horizon and return the correct value iteration.
	Horizon *h = lpomdp->get_horizon();
	if (h->is_finite()) {
		throw CoreException();
	}

	return solve_infinite_horizon(S, A, Z, T, O, R, h, lpomdp->get_slack());
}

PolicyAlphaVectors **LPBVI::solve_infinite_horizon(StatesMap *S, ActionsMap *A,
		ObservationsMap *Z, StateTransitions *T, ObservationTransitions *O,
		FactoredRewards *R, Horizon *h, std::vector<float> &delta)
{
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

	// Before anything, cache Gamma_{a, *} for all actions, but one for each R[i] now. This is used in every
	// cross-sum computation, but it's alright that this doesn't depend on b and is over all actions, since we
	// only ever use the ones with the action specified in the map. Since the inner loop only iterates over Ai[b]
	// actions, it naturally restricts the actions.
	std::map<Action *, std::vector<PolicyAlphaVector *> > *gammaAStar =
			new std::map<Action *, std::vector<PolicyAlphaVector *> >[R->get_num_rewards()];
	for (unsigned int i = 0; i < R->get_num_rewards(); i++) {
		SARewards *Ri = dynamic_cast<SARewards *>(R->get(i));
		if (Ri == nullptr) {
			throw RewardException();
		}

		for (auto a : *A) {
			Action *action = resolve(a);
			gammaAStar[i][action].push_back(create_gamma_a_star(S, Z, T, O, Ri, action));
		}
	}

	// If we are recording a belief point's values, create the empty vector for each R[i].
	if (beliefToRecord != nullptr) {
		recordedValues.resize(R->get_num_rewards());
	}

	// Perform a predefined number of expansions. Each update adds more belief points to the set B.
	for (unsigned int e = 0; e < expansions; e++) {
		std::cout << "Expansion " << (e + 1) << std::endl;

		// Create the set of actions available, one for each belief point; it starts with all actions available.
		std::map<BeliefState *, std::vector<Action *> > Ai;
		for (BeliefState *b : B) {
			for (auto a : *A) {
				Ai[b].push_back(resolve(a));
			}
		}

		// Compute the density of the belief points.
		double deltaB = compute_belief_density(S);

		// Actually run the bellman updates for each reward in sequence.
		for (unsigned int i = 0; i < R->get_num_rewards(); i++) {
			std::cout << "  R[" << i << "]" << std::endl; std::cout.flush();

			SARewards *Ri = dynamic_cast<SARewards *>(R->get(i));

			// Create the set of alpha vectors, which we call Gamma. As well as the previous Gamma set.
			std::vector<PolicyAlphaVector *> gamma[2];
			bool current = false;

			// Initialize the first set Gamma to be a set of zero alpha vectors.
			for (unsigned int j = 0; j < B.size(); j++) {
				PolicyAlphaVector *zeroAlphaVector = new PolicyAlphaVector();
				for (auto s : *S) {
//					zeroAlphaVector->set(resolve(s), Ri->get_min() / (1.0 - h->get_discount_factor()));
					zeroAlphaVector->set(resolve(s), 0.0);
				}
				gamma[!current].push_back(zeroAlphaVector);
			}

			// Perform a predefined number of updates. Each update improves the value function estimate.
			for (unsigned int u = 0; u < updates; u++) {
				std::cout << "    " << (u + 1) << " / " << updates << std::endl; std::cout.flush();

				// For each of the belief points, we must compute the optimal alpha vector.
//				int beliefCounter = 0;
				for (BeliefState *belief : B) {
//					std::cout << "      " << (beliefCounter + 1) << " / " << B.size() << std::endl; std::cout.flush();
//					beliefCounter++;

					PolicyAlphaVector *maxAlphaB = nullptr;
					double maxAlphaDotBeta = 0.0;

					// Compute the optimal alpha vector for this belief state.
//					int actionCounter = 0;
					for (Action *action : Ai[belief]) {
//						std::cout << "        " << (actionCounter + 1) << " / " << Ai[belief].size() << std::endl; std::cout.flush();
//						actionCounter++;

						PolicyAlphaVector *alphaBA = bellman_update_belief_state(S, Z, T, O, h,
								gammaAStar[i][action], gamma[!current], action, belief);

						double alphaDotBeta = alphaBA->compute_value(belief);
						if (maxAlphaB == nullptr || alphaDotBeta > maxAlphaDotBeta) {
							// This is the maximal alpha vector, so delete the old one.
							if (maxAlphaB != nullptr) {
								delete maxAlphaB;
							}
							maxAlphaB = alphaBA;
							maxAlphaDotBeta = alphaDotBeta;
						} else {
							// This was not the maximal alpha vector, so delete it.
							delete alphaBA;
						}
					}

					gamma[current].push_back(maxAlphaB);
				}

				// If we are recording values, compute the belief value here.
				if (beliefToRecord != nullptr) {
					double maxRecordedValue = std::numeric_limits<double>::lowest();
					for (PolicyAlphaVector *alphaRecord : gamma[current]) {
						double recordedValue = alphaRecord->compute_value(beliefToRecord);
						if (recordedValue > maxRecordedValue) {
							maxRecordedValue = recordedValue;
						}
					}
					recordedValues[i].push_back(maxRecordedValue);
				}

				// Prepare the next time step's gamma by clearing it. Remember again, we don't free the memory
				// because policy manages the previous time step's gamma (above). If this is the first horizon,
				// however, we actually do need to clear the set of zero alpha vectors.
				for (PolicyAlphaVector *zeroAlphaVector : gamma[!current]) {
					delete zeroAlphaVector;
				}
				gamma[!current].clear();
				current = !current;
			}

			// Set the current gamma to the policy object. Note: This transfers the responsibility of
			// memory management to the PolicyAlphaVectors object.
			policy[i]->set(gamma[!current]);

			// Setup the one-step slack eta_i value.
			double etai = delta[i];

			if (constrainEta) {
				double epsiloni = (Ri->get_max() - Ri->get_min()) / (1.0 - h->get_discount_factor()) * deltaB;
				etai = std::max(0.0, (1.0 - h->get_discount_factor()) * delta[i] - epsiloni);
			}

			// Restrict the set of actions available to each belief point in the next i+1 value function.
			if (i < R->get_num_rewards() - 1) {
				for (BeliefState *b : B) {
					policy[i]->get(b, etai, Ai[b]);
//					std::cout << "Ai[b].size() = " << Ai[b].size() << std::endl; std::cout.flush();
				}
			}

//			std::cout << "delta[i] = " << delta[i] << std::endl; std::cout.flush();
//			std::cout << "etai = " << etai << std::endl; std::cout.flush();
//			std::cout << "epsiloni = " << epsiloni << std::endl; std::cout.flush();
		}

		// Perform an expansion based on the rule the user wishes to use.
		if (e < expansions - 1) {
			switch (rule) {
			case POMDPPBVIExpansionRule::NONE:
				e = expansions; // Stop immediately if the user does not want to expand.
				break;
			case POMDPPBVIExpansionRule::RANDOM_BELIEF_SELECTION:
				expand_random_belief_selection(S);
				break;
			case POMDPPBVIExpansionRule::STOCHASTIC_SIMULATION_RANDOM_ACTION:
				expand_stochastic_simulation_random_actions(S, A, Z, T, O);
				break;
	//		case POMDPPBVIExpansionRule::STOCHASTIC_SIMULATION_GREEDY_ACTION:
				// NOTE: This one is a bit harder, since gamma is inside another loop now, but this is outside
				// that loop... Just ignore it for now, and use the one below.
	//			expand_stochastic_simulation_greedy_action(S, A, Z, T, O, gamma[!current]);
	//			break;
			case POMDPPBVIExpansionRule::STOCHASTIC_SIMULATION_EXPLORATORY_ACTION:
				expand_stochastic_simulation_exploratory_action(S, A, Z, T, O);
				break;
			case POMDPPBVIExpansionRule::GREEDY_ERROR_REDUCTION:
				expand_greedy_error_reduction();
				break;
			default:
				throw PolicyException();
				break;
			};
		}
	}

	// Free the memory of Gamma_{a, *}.
	for (unsigned int i = 0; i < R->get_num_rewards(); i++) {
		for (auto a : *A) {
			Action *action = resolve(a);
			for (PolicyAlphaVector *alphaVector : gammaAStar[i][action]) {
				delete alphaVector;
			}
			gammaAStar[i][action].clear();
		}
		gammaAStar[i].clear();
	}
	delete [] gammaAStar;

	return policy;
}

double LPBVI::compute_belief_density(StatesMap *S)
{
	double density = 0.0;

	for (BeliefState *bPrime : B) {
		for (BeliefState *b : B) {
			for (auto s : *S) {
				State *state = resolve(s);
				density = std::max(density, std::fabs(b->get(state) - bPrime->get(state)));
			}
		}
	}

	return density;
}

void LPBVI::reset()
{
	if (beliefToRecord != nullptr) {
		delete beliefToRecord;
	}
	beliefToRecord = nullptr;
	recordedValues.clear();
}
