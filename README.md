departHMM
=========

v1.0 of code to infer departures from routine in daily life mobility.

Runs EM algorithm to infer latent variables indicating: (0) user is in routine, or (1) user is departing from routine. Since we use a probabilistic approach, the inferred state of the user at each time step can take any value [0,1] indicating different degrees of departure.

What the code does:

0) generates synthetic data

1) runs the Baum-Welch EM algorithm and plots the convergence properties

2) re-runs 2 times more, to check for other possible maxima on the data log-likelihood surface

3) plots the inferred latent variables indicating departure (blue crosses) and the ground truth (red dots). Also plots the inferred trajectory using the Viterbi algorithm (green vertical dashes).