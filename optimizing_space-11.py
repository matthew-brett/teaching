# Make slice full of zeros like mid_vol1 slice
shifted_mid_vol1 = np.zeros(mid_vol1.shape)
# Fill the lower 54 (of 64) x lines with mid_vol1
shifted_mid_vol1[8:, :] = mid_vol1[:-8, :]
# Now we have something like mid_vol1 but translated
# in the first dimension
plt.imshow(shifted_mid_vol1)
# <...>
