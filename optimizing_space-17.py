# Number of voxels in the image
n_voxels = np.prod(mid_vol1.shape)
# Reshape vol0 slice as 1D vector
mid_vol0_as_1d = mid_vol0.reshape(n_voxels)
# Reshape vol1 slice as 1D vector
mid_vol1_as_1d = mid_vol1.reshape(n_voxels)
# These original slices should be very close to each other already
# So - plotting one set of image values against the other should
# be close to a straight line
plt.plot(mid_vol0_as_1d, mid_vol1_as_1d, '.')
# [...]
plt.xlabel('voxels in vol0 slice')
# <...>
plt.ylabel('voxels in original vol1 slice')
# <...>
# Correlation coefficient between them
print(np.corrcoef(mid_vol0_as_1d, mid_vol1_as_1d)[0, 1])
# 0.998119433113
