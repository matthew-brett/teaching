X_both_o = np.vstack((hrf1, unique_hrf2, np.ones_like(hrf1))).T
plt.imshow(X_both_o, interpolation='nearest', cmap='gray')
# <...>
