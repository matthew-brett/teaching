# Formula as a function we can evaluate
post_prob_bias_func = lambdify((alpha, beta, pi, u), post_prob_bias)
pi_vals = np.linspace(0, 1, 100)
fig, axes = plt.subplots(3, 1, figsize=(8, 16))
for i, power in enumerate((0.8, 0.5, 0.2)):
    beta_val = 1 - power
    for bias in (0.05, 0.2, 0.5, 0.8):
        pp_vals = post_prob_bias_func(0.05, beta_val, pi_vals, bias)
        axes[i].plot(pi_vals, pp_vals, label='u={0}'.format(bias))
        axes[i].set_ylabel('Posterior probability $Pr(H_A | T_S)$')
    axes[i].set_title('Power = {0}'.format(power))
    axes[-1].set_xlabel('Prior probability $Pr(H_A)$')
axes[-1].legend()