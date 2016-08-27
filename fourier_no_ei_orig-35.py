w_again = np.zeros(w.shape, dtype=np.complex)
c_0 = np.ones(N)
for n in np.arange(N):
    w_again[n] = 1. / N * c_0.dot(W)
w_again
# array([ 2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j,
# 2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j,
# 2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j,
# 2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j,
# 2.+0.j,  2.+0.j,  2.+0.j,  2.+0.j])
