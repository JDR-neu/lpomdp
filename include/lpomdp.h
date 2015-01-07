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


#ifndef LPOMDP_H
#define LPOMDP_H


#include "../../librbr/librbr/include/pomdp/pomdp.h"

#include "../../librbr/librbr/include/core/rewards/factored_rewards.h"

/**
 * A MOPOMDP with lexicographic reward preferences which allows for slack.
 */
class LPOMDP : public POMDP {
public:
	/**
	 * The default constructor for the LPOMDP class.
	 */
	LPOMDP();

	/**
	 * A constructor for the LPOMDP class.
	 * @param	S		The states.
	 * @param	A		The actions.
	 * @param	Z		The observations.
	 * @param	T		The state transitions, which uses the states and actions parameters.
	 * @param	O		The observation transitions, which uses the states, actions, and observations parameters.
	 * @param	R		The rewards, which uses the states and actions parameters.
	 * @param	h		The horizon.
	 * @param	d		The slack vector of size k. Values must be non-negative.
	 */
	LPOMDP(States *S, Actions *A, Observations *Z, StateTransitions *T, ObservationTransitions *O,
			FactoredRewards *R, Initial *s, Horizon *h, std::vector<float> *d);

	/**
	 * The default deconstructor for the LPOMDP class.
	 */
	virtual ~LPOMDP();

	/**
	 * Get the factored rewards. This is an overloaded method, allowing for explicit
	 * return of a FactoredRewards object.
	 * @return	The factored rewards.
	 */
	FactoredRewards *get_rewards();

	/**
	 * Set the slack.
	 * @param	d	The new slack.
	 */
	void set_slack(const std::vector<float> &d);

	/**
	 * Get the slack.
	 * @return	The slack vector.
	 */
	std::vector<float> &get_slack();

protected:
	/**
	 * The slack as a k-array; each element must be non-negative.
	 */
	std::vector<float> delta;

};


#endif // LPOMDP_H
