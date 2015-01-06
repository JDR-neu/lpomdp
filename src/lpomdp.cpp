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


#include "../include/lpomdp.h"

#include "../../librbr/librbr/include/core/rewards/reward_exception.h"

LPOMDP::LPOMDP()
{ }

LPOMDP::LPOMDP(States *S, Actions *A, Observations *Z, StateTransitions *T, ObservationTransitions *O,
		FactoredRewards *R, Initial *s, Horizon *h, std::vector<float> *d) : POMDP(S, A, Z, T, O, R, h)
{
	for (float val : *d) {
		delta.push_back(val);
	}
}

LPOMDP::~LPOMDP()
{
	delta.clear();
}

FactoredRewards *LPOMDP::get_rewards()
{
	FactoredRewards *R = dynamic_cast<FactoredRewards *>(rewards);
	if (R == nullptr) {
		throw RewardException();
	}
	return R;
}

void LPOMDP::set_slack(const std::vector<float> &d)
{
	delta.clear();
	for (float val : d) {
		delta.push_back(val);
	}
}

std::vector<float> &LPOMDP::get_slack()
{
	return delta;
}
