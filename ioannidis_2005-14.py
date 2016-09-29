# Formula as a function we can evaluate
post_prob_multi_func = lambdify((alpha, beta, pi, n), post_prob_multi)
pi_vals = np.linspace(0, 1, 100)
fig, axes = plt.subplots(3, 1, figsize=(8, 16))
for i, power in enumerate((0.8, 0.5, 0.2)):
    beta_val = 1 - power
    for n_studies in (1, 5, 10, 50):
        pp_vals = post_prob_multi_func(0.05, beta_val, pi_vals, n_studies)
        axes[i].plot(pi_vals, pp_vals, label='n={0}'.format(n_studies))
        axes[i].set_ylabel('Posterior probability $Pr(H_A | T_S)$')
    axes[i].set_title('Power = {0}'.format(power))
axes[-1].set_xlabel('Prior probability $Pr(H_A)$')
axes[-1].legend()