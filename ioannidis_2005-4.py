# Load libraries for making plot
import numpy as np
import matplotlib.pyplot as plt
# Make symbolic expression into a function we can evaluate
post_prob_func = lambdify((alpha, beta, pi), post_prob)
# Make set of pi values for x axis of plot
pi_vals = np.linspace(0, 1, 100)
for power in (0.8, 0.5, 0.2):
    beta_val = 1 - power
    plt.plot(pi_vals, post_prob_func(0.05, beta_val, pi_vals),
             label='power={0}'.format(power))
plt.xlabel('Prior probability $Pr(H_A)$')
plt.ylabel('Posterior probability $Pr(H_A | T_S)$')
plt.legend()
plt.title("Posterior probability for different priors and power levels")