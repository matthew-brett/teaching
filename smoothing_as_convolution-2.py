np.random.seed(5) # To get predictable random numbers
n_points = 40
x_vals = np.arange(n_points)
y_vals = np.random.normal(size=n_points)
plt.bar(x_vals, y_vals)
# <...>
