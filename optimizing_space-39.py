best_params = fmin_powell(cost_at_xyz, [0, 0, 0], callback=my_callback)
# Trying parameters [-7.5722 -4.9661  0.    ]
# Trying parameters [-7.9968 -4.9997  0.    ]
# Trying parameters [-8. -5.  0.]
# Optimization terminated successfully.
# Current function value: -0.995145
# Iterations: 3
# Function evaluations: 270
best_params
# array([-8., -5.,  0.])
