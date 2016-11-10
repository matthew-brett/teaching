plt.plot(t1_slice.ravel(), t2_slice.ravel(), '.')
# [...]
plt.xlabel('T1 signal')
# <...>
plt.ylabel('T2 signal')
# <...>
plt.title('T1 vs T2 signal')
# <...>
np.corrcoef(t1_slice.ravel(), t2_slice.ravel())[0, 1]
# 0.787079...
