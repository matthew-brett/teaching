# Regress hrf2 against hrf1 to get best fit of hrf2 using just hrf1
y = hrf2
X = hrf1[:, np.newaxis]  # hrf1 as column vector
B_hrf1_in_hrf2 = npl.pinv(X).dot(y)  # scalar multiple of hrf1 to best fit hrf2
hrf1_in_hrf2 = X.dot(B_hrf1_in_hrf2)  # portion of hrf2 that can be explained by hrf1
unique_hrf2 = hrf2 - hrf1_in_hrf2  # portion of hrf2 that cannot be explained by hrf1
plt.plot(times, hrf1, label='hrf1')
# [...]
plt.plot(times, hrf2, label='hrf2')
# [...]
plt.plot(times, hrf1_in_hrf2, label='hrf1 in hrf2')
# [...]
plt.plot(times, unique_hrf2, label='hrf2 orth wrt hrf1')
# [...]
plt.legend()
# <...>
# hrf1 part of hrf2, plus unique part, equals original hrf2
np.allclose(hrf2, hrf1_in_hrf2 + unique_hrf2)
# True
