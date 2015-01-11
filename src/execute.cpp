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

	losmLPOMDP->set_slack(10.0f, 0.0f);

	LPBVI solver;
	solver.set_expansion_rule(POMDPPBVIExpansionRule::STOCHASTIC_SIMULATION_EXPLORATORY_ACTION);
	/*
	solver.set_num_expansion_iterations(1); // No expansions!
	solver.compute_num_update_iterations(losmLPOMDP, 1.0); // Within 1.0 of optimal answer!
	solver.set_num_update_iterations(solver.get_num_update_iterations() / 1);
	//*/

	//*
	solver.set_num_expansion_iterations(1);
	solver.set_num_update_iterations(100);
	//*/

	/*
	solver.set_num_expansion_iterations(10);
	solver.set_num_update_iterations(100);
	//*/



	// TODO: You need to assign which actions are available at each state (yes state) so that you can remove
	// the "-1e+35" penalty for taking impossible actions at intersections. The reason is that it screws up
	// the calculation of epsiloni in lpbvi.cpp, which is subtracted from the val to get etai....

	// TODO: Once you fix above, also in lpbvi.cpp at the initialization of zeroAlphaVector, use the
	// R_min / (1 - gamma) instead of 0.



	// Add all states as perfect belief points; this is similar to an LMDP then.
	/*
	StatesMap *S = dynamic_cast<StatesMap *>(losmLPOMDP->get_states());
	for (auto s : *S) {
		LOSMState *state = dynamic_cast<LOSMState *>(resolve(s));
		BeliefState *b = new BeliefState();
		b->set(state, 1.0);
		solver.add_initial_belief_state(b);
	}
	//*/

	// Add uniform distribution over tiredness possibilities of states as belief points.
	//*
	for (auto statesVector : losmLPOMDP->get_tiredness_states()) {
		BeliefState *b = new BeliefState();
		for (LOSMState *state : statesVector) {
			b->set(state, 1.0 / (double)statesVector.size());
		}
		solver.add_initial_belief_state(b);
	}
	//*/

	// Add combinations of goal states as the belief points. We will instead run some expansions off of these.
	/*
	for (LOSMState *s : losmLPOMDP->get_goal_states()) {
		BeliefState *b = new BeliefState();
		b->set(s, 1.0);
		solver.add_initial_belief_state(b);
	}
	//*/

	PolicyAlphaVectors **policy = nullptr;
//	try {
		policy = solver.solve(losmLPOMDP);
//	} catch (const CoreException &err) {
//		std::cout << " Failure." << std::endl;
//	} catch (const StateException &err) {
//		std::cout << " Failure." << std::endl;
//	} catch (const ActionException &err) {
//		std::cout << " Failure." << std::endl;
//	} catch (const ObservationException &err) {
//		std::cout << " Failure." << std::endl;
//	} catch (const StateTransitionException &err) {
//		std::cout << " Failure." << std::endl;
//	} catch (const ObservationTransitionException &err) {
//		std::cout << " Failure." << std::endl;
//	} catch (const RewardException &err) {
//		std::cout << " Failure." << std::endl;
//	} catch (const PolicyException &err) {
//		std::cout << " Failure." << std::endl;
//	}
		losmLPOMDP->save_policy(policy, losmLPOMDP->get_rewards()->get_num_rewards(), argv[8]);

	for (unsigned int i = 0; i < losmLPOMDP->get_rewards()->get_num_rewards(); i++) {
		delete [] policy[i];
	}
	delete [] policy;

	return 0;
}