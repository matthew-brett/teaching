# Use the pseudoinverse to get estimated B
B = npl.pinv(X).dot(psychopathy)
B
# array([ 10.071286,   0.999257])

# Plot the data
plt.plot(clammy, psychopathy, '+')
# [...]

def my_best_line(x):
    # Best prediction for psychopathy given clamminess
    return B[0] + B[1] * x

# Plot the new prediction
x_vals = [0, max(clammy)]
y_vals = [my_best_line(0), my_best_line(max(clammy))]
plt.plot(x_vals, y_vals)
# [...]
plt.xlabel('Clamminess of handshake')
# <...>
plt.ylabel('Psychopathy score')
# <...>
