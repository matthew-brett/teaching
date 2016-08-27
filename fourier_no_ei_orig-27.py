# Does it actually work?
unique_Cs = C[:N/2+1, :]
unique_Ss = S[1:N/2, :]
small_n = len(unique_Ss)
cos_dots = unique_Cs.dot(x)
sin_dots = unique_Ss.dot(x)
cos_gs = cos_dots / ([N] + [N/2] * small_n + [N])
sin_gs = sin_dots / ([N/2] * small_n)
cos_projections = cos_gs[:, None] * unique_Cs
sin_projections = sin_gs[:, None] * unique_Ss
x_back = np.sum(np.vstack((cos_projections, sin_projections)), axis=0)
x_back - x
# array([-0.,  0., -0.,  0., -0., -0.,  0., -0.,  0., -0.,  0., -0.,  0.,
# -0., -0.,  0., -0.,  0., -0., -0.,  0.,  0., -0., -0.,  0., -0.,
# 0.,  0.,  0., -0.,  0., -0.])
