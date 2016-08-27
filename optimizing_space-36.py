shifted_vol1 = np.zeros(vol1.shape)
shifted_vol1[8:, 5:, :] = vol1[:-8, :-5, :]
plt.imshow(shifted_vol1[:, :, 17])
# <...>
