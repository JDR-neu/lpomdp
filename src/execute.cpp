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

	// Load the LOSM POMDP.
	LOSMPOMDP *losmPOMDP = nullptr;
	try {
		losmPOMDP = new LOSMPOMDP(argv[1], argv[2], argv[3], argv[6], argv[7]);
	} catch (LOSMException &err) {
		std::cerr << "Failed to load the files provided." << std::endl;
		return -1;
	}

	std::cout << "1"; std::cout.flush();
	losmPOMDP->set_slack(10.0f, 0.0f);

	unsigned int numExpansions = 1;

	std::cout << "2"; std::cout.flush();
	LPBVI solver;
	solver.set_expansion_rule(POMDPPBVIExpansionRule::STOCHASTIC_SIMULATION_EXPLORATORY_ACTION);
	solver.set_num_expansion_iterations(numExpansions); // No expansions!
	solver.compute_num_update_iterations(losmPOMDP, 1.0); // Within 1.0 of optimal answer!
	solver.set_num_update_iterations(solver.get_num_update_iterations() / numExpansions);

	// Add all states as belief points.
	//*
	StatesMap *S = dynamic_cast<StatesMap *>(losmPOMDP->get_states());
	for (auto s : *S) {
		LOSMState *state = dynamic_cast<LOSMState *>(resolve(s));
		BeliefState *b = new BeliefState();
		b->set(state, 1.0);
		solver.add_initial_belief_state(b);
	}
	//*/

	// Add the initial state as the belief point. We will instead run some expansions.
	/*
	// TODO!!!
	//*/

	PolicyAlphaVectors **policy = nullptr;
	std::cout << "3"; std::cout.flush();
//	try {
		policy = solver.solve(losmPOMDP);
		std::cout << "4"; std::cout.flush();
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
	losmPOMDP->save_policy(policy, losmPOMDP->get_rewards()->get_num_rewards(), argv[8]);
	std::cout << "5"; std::cout.flush();

	for (unsigned int i = 0; i < losmPOMDP->get_rewards()->get_num_rewards(); i++) {
		delete [] policy[i];
	}
	delete [] policy;

	return 0;
}
