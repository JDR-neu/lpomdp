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


#ifndef LPBVI_H
#define LPBVI_H


#include "lpomdp.h"

#include "../../librbr/librbr/include/pomdp/pomdp_pbvi.h"

#include "../../librbr/librbr/include/core/policy/policy_alpha_vectors.h"

#include "../../librbr/librbr/include/core/states/states_map.h"
#include "../../librbr/librbr/include/core/actions/actions_map.h"
#include "../../librbr/librbr/include/core/observations/observations_map.h"
#include "../../librbr/librbr/include/core/state_transitions/state_transitions.h"
#include "../../librbr/librbr/include/core/observation_transitions/observation_transitions.h"
#include "../../librbr/librbr/include/core/rewards/factored_rewards.h"
#include "../../librbr/librbr/include/core/rewards/sa_rewards.h"
#include "../../librbr/librbr/include/core/initial.h"
#include "../../librbr/librbr/include/core/horizon.h"

#include <unordered_map>

/**
 * Solve a Lexicographic Partially Observable Markov Decision Process (LMDP).
 */
class LPBVI : public POMDPPBVI {
public:
	/**
	 * The default constructor for the LPBVI class. Default number of iterations for infinite
	 * horizon POMDPs is 1. The default expansion rule is Random Belief Selection.
	 */
	LPBVI();

	/**
	 * A constructor for the LPBVI class which allows for the specification of the expansion rule,
	 * and the number of iterations (both updates and expansions) to run for infinite horizon.
	 * The default is 1 for both.
	 * @param	expansionRule			The expansion rule to use.
	 * @param	updateIterations 		The number of update iterations to run for infinite horizon POMDPs.
	 * @param	expansionIterations 	The number of expansion iterations to run for infinite horizon POMDPs.
	 */
	LPBVI(POMDPPBVIExpansionRule expansionRule, unsigned int updateIterations, unsigned int expansionIterations);

	/**
	 * The deconstructor for the LPBVI class. This method frees all the belief state memory.
	 */
	virtual ~LPBVI();

	/**
	 * Compute the optimal number of update iterations to run for infinite horizon POMDPs, given
	 * the desired tolerance, requiring knowledge of the reward function. This selects the maximum over
	 * all of the rewards.
	 * @param	pomdp 				The partially observable Markov decision process to use. Must be an LPOMDP.
	 * @param	epsilon				The desired tolerance between value functions to check for convergence.
	 * @throw	RewardException		The POMDP did not have a SARewards rewards object.
	 */
	void compute_num_update_iterations(POMDP *pomdp, double epsilon);

	/**
	 * Setup the belief state to record over iterations.
	 * @param	b	The belief state to record.
	 */
	void set_belief_to_record(BeliefState *b);

	/**
	 * Get the recorded values of the belief state for all value functions. These values are set
	 * during both the "solve" function and
	 * @return	The vector over time of each vector of values of the specified belief state.
	 */
	const std::vector<std::vector<double> > &get_recorded_values() const;

	/**
	 * Whether or not to constrain eta for theoretical guarantee.
	 * @param	value	Constraint it or not.
	 */
	void eta_constraint(bool value);

	/**
	 * Throw an error if they try to solve just a POMDP.
	 * @param	pomdp				The partially observable Markov decision process to solve.
	 * @throw	CoreException		This is a POMDP.
	 * @return	Return null.
	 */
	PolicyAlphaVectors *solve(POMDP *pomdp);

	/**
	 * Solve the LPOMDP provided using lexicographic point-based value iteration.
	 * @param	pomdp							The LPOMDP to solve.
	 * @throw	StateException					The LPOMDP did not have a StatesMap states object.
	 * @throw	ActionException					The LPOMDP did not have a ActionsMap actions object.
	 * @throw	ObservationException			The LPOMDP did not have a ObservationsMap actions object.
	 * @throw	StateTransitionsException		The LPOMDP did not have a StateTransitions state transitions object.
	 * @throw	ObservationTransitionsException	The LPOMDP did not have a ObservationTransitions observation transitions object.
	 * @throw	RewardException					The LPOMDP did not have a FactoredRewards (elements SARewards) rewards object.
	 * @throw	CoreException					The LPOMDP was not infinite horizon.
	 * @throw	PolicyException					An error occurred computing the policy.
	 * @return	Return the optimal policy, one set of alpha vectors for each value function.
	 */
	PolicyAlphaVectors **solve(LPOMDP *lpomdp);

	/**
	 * Compute the value of the belief states given a policy.
	 * @param	pomdp							The LPOMDP to solve.
	 * @param	policy							The policy mapping beliefs on the simplex to actions via their values.
	 * @throw	StateException					The LPOMDP did not have a StatesMap states object.
	 * @throw	ActionException					The LPOMDP did not have a ActionsMap actions object.
	 * @throw	ObservationException			The LPOMDP did not have a ObservationsMap actions object.
	 * @throw	StateTransitionsException		The LPOMDP did not have a StateTransitions state transitions object.
	 * @throw	ObservationTransitionsException	The LPOMDP did not have a ObservationTransitions observation transitions object.
	 * @throw	RewardException					The LPOMDP did not have a FactoredRewards (elements SARewards) rewards object.
	 * @throw	CoreException					The LPOMDP was not infinite horizon.
	 * @throw	PolicyException					An error occurred computing the policy.
	 * @return	Return the alpha values for the policy provided.
	 */
	PolicyAlphaVectors **compute_value(LPOMDP *lpomdp, PolicyAlphaVectors *policy);

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
	 * Compute the value of a policy at the belief points.
	 * @param	S					The finite states.
	 * @param	A					The finite actions.
	 * @param	Z					The finite observations.
	 * @param	T					The finite state transition function.
	 * @param	O					The finite observation transition function.
	 * @param	R					The factored state-action rewards.
	 * @param	h					The horizon.
	 * @param	delta				The slack vector.
	 * @param	policy				The policy mapping beliefs on the simplex to actions via their values.
	 * @throw	PolicyException		An error occurred computing the policy.
	 * @return	Return the optimal policy.
	 */
	virtual PolicyAlphaVectors **compute_value_execute(StatesMap *S, ActionsMap *A,
			ObservationsMap *Z, StateTransitions *T, ObservationTransitions *O,
			FactoredRewards *R, Horizon *h, std::vector<float> &delta,
			PolicyAlphaVectors *policy);

	/**
	 * Compute the approximate density (an upper bound) of the belief points.
	 * @param	S	The set of states.
	 * @return	The approximate density (an upper bound).
	 */
	virtual double compute_belief_density(StatesMap *S);

	/**
	 * Reset the internal variables.
	 */
	virtual void reset();

	/**
	 * The belief state to record.
	 */
	BeliefState *beliefToRecord;

	/**
	 * The recorded values of the belief state above over iteration for each value function.
	 */
	std::vector<std::vector<double> > recordedValues;

	/**
	 * Whether or not to constrain eta for theoretical guarantees.
	 */
	bool constrainEta;

};


#endif // LVI_H
