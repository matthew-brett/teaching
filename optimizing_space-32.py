best_params = fmin_powell(cost_function, [0], callback=my_callback)
# Trying parameters [-7.995]
# Trying parameters [-7.996]
# Optimization terminated successfully.
# Current function value: -0.997032
# Iterations: 2
# Function evaluations: 27
print(best_params)
# -7.9960266912...
