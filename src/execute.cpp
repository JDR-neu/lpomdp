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

#include "../include/losm_lpomdp.h"
#include "../include/lpbvi.h"
#include "../include/lpbvi_cuda.h"

#include "../../losm/losm/include/losm_exception.h"

#include <iostream>
#include <chrono>

int main(int argc, char *argv[])
{
	// Ensure the correct number of arguments.
	if (argc != 9) {
		std::cerr << "Please specify nodes, edges, and landmarks data files, as well as the initial and goal nodes' UIDs, plus the policy output file." << std::endl;
		return -1;
	}

	// Load the LOSM LPOMDP.
	LOSMPOMDP *losmLPOMDP = nullptr;
	try {
		losmLPOMDP = new LOSMPOMDP(argv[1], argv[2], argv[3], argv[6], argv[7]);
	} catch (LOSMException &err) {
		std::cerr << "Failed to load the files provided." << std::endl;
		return -1;
	}

	losmLPOMDP->set_slack(30.0f, 0.0f);

	// -------------------------------------------------------------------------------------
	//* CPU Version
	LPBVI solver;
	solver.set_num_update_iterations(10);
	//*/

	/* GPU Version
	LPBVICuda solver;
	solver.set_performance_variables(2, 2);
	solver.set_num_update_iterations(500);
	//*/
	// -------------------------------------------------------------------------------------

	solver.eta_constraint(false);
	solver.set_expansion_rule(POMDPPBVIExpansionRule::STOCHASTIC_SIMULATION_EXPLORATORY_ACTION);
	solver.set_num_expansion_iterations(1);



	// TODO: You need to assign which actions are available at each state (yes state) so that you can remove
	// the "-1e+35" penalty for taking impossible actions at intersections. The reason is that it screws up
	// the calculation of epsiloni in lpbvi.cpp, which is subtracted from the val to get etai....

	// TODO: Once you fix above, also in lpbvi.cpp at the initialization of zeroAlphaVector, use the
	// R_min / (1 - gamma) instead of 0.

	// TODO: You need to compute the V^\pi value, and plot that. Currently, you are using the V^\eta values,
	// which are detached from one another, namely the first one. Once you compute the correct value you
	// will again be able to numerically see the effect of slack.


	// Find the belief to record given the two initial UIDs defining the initial state.
	BeliefState *beliefToRecord = nullptr;
	unsigned long uid1 = std::stol(argv[4]);
	unsigned long uid2 = std::stol(argv[5]);

	for (auto statesVector : losmLPOMDP->get_tiredness_states()) {
		// All the states in a statesVector are constructed with the same pair of UIDs;
		// the only difference is the tiredness values.
		LOSMState *s1 = statesVector[0];
		if ((s1->get_current()->get_uid() == uid1 && s1->get_previous()->get_uid() == uid2) ||
				(s1->get_current()->get_uid() == uid2 && s1->get_previous()->get_uid() == uid1)) {
			beliefToRecord = new BeliefState();
			beliefToRecord->set(statesVector[0], 0.5);
			beliefToRecord->set(statesVector[1], 0.5);
			solver.set_belief_to_record(beliefToRecord);

			std::cout << "Found and set a belief to record." << std::endl; std::cout.flush();
			break;
		}
	}

	// Add uniform distribution over tiredness possibilities of states as belief points.
	//*
	for (auto statesVector : losmLPOMDP->get_tiredness_states()) {
		// The size of statesVector is always 2 in our case.
		BeliefState *b = nullptr;

		b = new BeliefState();
		b->set(statesVector[0], 1.0);
		b->set(statesVector[1], 0.0);
		solver.add_initial_belief_state(b);

		b = new BeliefState();
		b->set(statesVector[0], 0.75);
		b->set(statesVector[1], 0.25);
		solver.add_initial_belief_state(b);

		b = new BeliefState();
		b->set(statesVector[0], 0.5);
		b->set(statesVector[1], 0.5);
		solver.add_initial_belief_state(b);

		b = new BeliefState();
		b->set(statesVector[0], 0.25);
		b->set(statesVector[1], 0.75);
		solver.add_initial_belief_state(b);

		b = new BeliefState();
		b->set(statesVector[0], 0.0);
		b->set(statesVector[1], 1.0);
		solver.add_initial_belief_state(b);
	}
	//*/

	PolicyAlphaVectors **policy = nullptr;
	policy = solver.solve(losmLPOMDP);
//	losmLPOMDP->save_policy(policy, losmLPOMDP->get_rewards()->get_num_rewards(), argv[8]);
	losmLPOMDP->save_policy(policy, losmLPOMDP->get_rewards()->get_num_rewards(), 0.20, argv[8]);

	// Print the V^eta of this belief state.
	std::cout << "V^eta(b^0): [" << policy[0]->compute_value(beliefToRecord) << ", " <<
			policy[1]->compute_value(beliefToRecord) << "]" << std::endl; std::cout.flush();

	// Free the belief to record value.
	delete beliefToRecord;
	beliefToRecord = nullptr;

	/* After everything is computed, output the recorded values in a csv-like format to the screen.
	std::cout << "V^eta(b^0):" << std::endl; std::cout.flush();
	for (auto Vi : solver.get_recorded_values()) {
		for (double Vit : Vi) {
			std::cout << Vit << ",";
		}
		std::cout << std::endl; std::cout.flush();
	}
	//*/

	/* We take the final collection of alpha vectors as the final policy, hence policy[1].
	PolicyAlphaVectors **result = nullptr;
	result = solver.compute_value(losmLPOMDP, policy[1]);

	// Now compute the actual values following the policy.
	std::cout << "V^pi(b^0):" << std::endl; std::cout.flush();
	for (auto Vi : solver.get_recorded_values()) {
		for (double Vit : Vi) {
			std::cout << Vit << ",";
		}
		std::cout << std::endl; std::cout.flush();
	}

	// Free the result memory.
	for (unsigned int i = 0; i < losmLPOMDP->get_rewards()->get_num_rewards(); i++) {
		delete [] result[i];
	}
	delete [] result;
	//*/

	// Free the policy memory.
	for (unsigned int i = 0; i < losmLPOMDP->get_rewards()->get_num_rewards(); i++) {
		delete [] policy[i];
	}
	delete [] policy;

	return 0;
}
