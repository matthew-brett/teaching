x_position = 13
# Make Gaussian centered at 13 with given sigma
kernel_at_pos = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
# Make kernel sum to 1
kernel_at_pos = kernel_at_pos / sum(kernel_at_pos)
plt.bar(x_vals, kernel_at_pos)
# <...>
