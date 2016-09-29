X = np.column_stack((np.ones(12), clammy))
Y = np.asarray(psychopathy)
B, t, df, p = t_stat(Y, X, [0, 1])
t, p
# (array([[ 1.914389]]), array([[ 0.042295]]))
