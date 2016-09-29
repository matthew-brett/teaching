X_both = np.vstack((hrf1, hrf2, np.ones_like(hrf1))).T
plt.imshow(X_both, interpolation='nearest', cmap='gray')
# <...>
plt.title('Model with both HRF regressors')
# <...>
