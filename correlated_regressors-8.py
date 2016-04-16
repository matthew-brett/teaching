X_one = np.vstack((hrf1, np.ones_like(hrf1))).T
plt.imshow(X_one, interpolation='nearest', cmap='gray')
# <...>
plt.title('Model with first HRF regressor only')
# <...>
