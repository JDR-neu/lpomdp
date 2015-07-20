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

#include "../../librbr/librbr/include/management/pomdp_file.h"

#include "../../losm/losm/include/losm_exception.h"

#include <iostream>
#include <chrono>

int main(int argc, char *argv[])
{
	// Ensure the correct number of arguments.
	if (argc != 9) {
		std::cerr << "Please specify nodes, edges, and landmarks data files, as well as the initial and goal nodes' UIDs, plus the Cassandra POMDP output file." << std::endl;
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

	// Find the belief to record given the two initial UIDs defining the initial state.
	BeliefState *start = nullptr;
	unsigned long uid1 = std::stol(argv[4]);
	unsigned long uid2 = std::stol(argv[5]);

	for (auto statesVector : losmLPOMDP->get_tiredness_states()) {
		// All the states in a statesVector are constructed with the same pair of UIDs;
		// the only difference is the tiredness values.
		LOSMState *s1 = statesVector[0];
		if ((s1->get_current()->get_uid() == uid1 && s1->get_previous()->get_uid() == uid2) ||
				(s1->get_current()->get_uid() == uid2 && s1->get_previous()->get_uid() == uid1)) {
			start = new BeliefState();
			start->set(statesVector[0], 0.5);
			start->set(statesVector[1], 0.5);

			std::cout << "Found and set the initial belief." << std::endl; std::cout.flush();
			break;
		}
	}

	std::cout << "Saving..." << std::endl;

	// Save the Cassandra POMDP file with the path and filename provided.
	POMDPFile file;
	file.save_pomdp(losmLPOMDP, argv[8], start);

	std::cout << "Done!" << std::endl;

	return 0;
}
