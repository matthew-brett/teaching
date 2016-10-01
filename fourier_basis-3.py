C = np.zeros((N, N))
S = np.zeros((N, N))
ns = np.arange(N)
one_cycle = 2 * np.pi * ns / N
for k in range(N):
    t_k = k * one_cycle
    C[k, :] = np.cos(t_k)
    S[k, :] = np.sin(t_k)
