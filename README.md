lpomdp
======

Lexicographic Partially Observable Markov Decision Processes (LPOMDPs) are MOPOMDPs with lexicographic preferences over the optimization of value functions, allowing for slack in their optimization. Lexicographic Value Iteration (LVI) for LMDPs solves this problem, but is too computationally expensive in practice. We include Lexicographic Point-Based Value Iteration (LPBVI) which approximates one of the solutions to the LPOMDP, but not necessarily the optimal solution to the problem.

For more information, please see our IJCAI 2015 paper:

Wray, Kyle H. and Zilberstein, Shlomo. "Multi-Objective POMDPs with Lexicographic Reward Preferences." In the Proceedings of the Twenty-Fourth International Joint Conference of Artificial Intelligence (IJCAI), Buenos Aires, Argentina, July 2015.
