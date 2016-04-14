smoothed_by_convolving = convolved_y[kernel_n_below_0:(n_points+kernel_n_below_0)]
plt.bar(x_vals, smoothed_by_convolving)
# <...>
