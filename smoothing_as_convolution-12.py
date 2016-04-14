# Multiply and sum y values within the finite kernel
kernel_starts_at = 13 - kernel_n_below_0
y_within_kernel = y_vals[kernel_starts_at : kernel_starts_at + len(finite_kernel)]
print(np.dot(finite_kernel, y_within_kernel))
# -0.347973672994
