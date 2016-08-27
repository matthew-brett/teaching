finite_kernel = kernel[kernel_above_thresh]
# Make kernel sum to 1 again
finite_kernel = finite_kernel / finite_kernel.sum()
plt.plot(x_within_thresh, finite_kernel)
# [...]
