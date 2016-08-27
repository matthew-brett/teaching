T = len(times)
iters = 10000
# Make 10000 Y vectors (new noise for each colum)
noise_vectors = np.random.normal(size=(T, iters))
# add signal to make data vectors
Ys = noise_vectors + signal[:, np.newaxis]
Ys.shape
# (15, 10000)
