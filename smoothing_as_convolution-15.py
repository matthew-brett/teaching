smoothed_vals = np.zeros(y_vals.shape)
for x_position in x_vals:
     kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
     kernel = kernel / sum(kernel)
     smoothed_vals[x_position] = sum(y_vals * kernel)
plt.bar(x_vals, smoothed_vals)
# <...>
