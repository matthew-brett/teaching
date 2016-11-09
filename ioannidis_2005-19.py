pp_vals_nobias = pp_mb_func(0.05, 0.92, pi_vals, 0, 1)
pp_vals_bias = pp_mb_func(0.05, 0.92, pi_vals, 0.1, 2)
plt.plot(pi_vals, pp_vals_nobias, label='no analysis or publication bias')
plt.plot(pi_vals, pp_vals_bias, label='with analysis and publication bias')
plt.plot(pi_vals, pi_vals, 'r:', label='$T_S$ not informative')
plt.ylabel('Posterior probability $Pr(H_A | T_S)$')
plt.xlabel('Prior probability $Pr(H_A)$')
plt.legend()