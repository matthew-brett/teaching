# Make a +/- x range large enough to let kernel drop to zero
x_for_kernel = np.arange(-10, 10)
# Calculate kernel
kernel = np.exp(-(x_for_kernel) ** 2 / (2 * sigma ** 2))
# Threshold
kernel_above_thresh = kernel > 0.0001
# Find x values where kernel is above threshold
x_within_thresh = x_for_kernel[kernel_above_thresh]
plt.plot(x_for_kernel, kernel)
# [...]
plt.plot(min(x_within_thresh), 0, marker=7, markersize=40)
# [...]
plt.plot(max(x_within_thresh), 0, marker=7, markersize=40)
# [...]
