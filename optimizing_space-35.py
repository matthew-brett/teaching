best_params = fmin_powell(fancy_cost_at_xy, [0, 0], callback=my_callback)
# Trying parameters [-7.995  0.   ]
# Trying parameters [-7.996  0.   ]
# Optimization terminated successfully.
# Current function value: -0.997032
# Iterations: 2
# Function evaluations: 133
best_params
# array([-7.996,  0.   ])
