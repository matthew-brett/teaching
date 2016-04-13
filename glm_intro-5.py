# Plot the data
plt.plot(clammy, psychopathy, '+')
# [...]

def my_line(x):
    # My prediction for psychopathy given clamminess
    return 10 + 0.9 * x

# Plot the prediction
x_vals = [0, max(clammy)]
y_vals = [my_line(0), my_line(max(clammy))]
plt.plot(x_vals, y_vals)
# [...]
plt.xlabel('Clamminess of handshake')
# <...>
plt.ylabel('Psychopathy score')
# <...>
plt.title('Clammy vs psychopathy with guessed line')
# <...>
