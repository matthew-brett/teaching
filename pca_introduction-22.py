angles = np.linspace(0, np.pi, 10000)
x = np.cos(angles)
y = np.sin(angles)
u_vectors = np.vstack((x, y))
u_vectors.shape
# (2, 10000)
