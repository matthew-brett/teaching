# The shifted slice will be less well matched
# Therefore the line will be less straight and narrow
plt.plot(mid_vol0_as_1d, shifted_mid_vol1.ravel(), '.')
# [...]
plt.xlabel('voxels in vol0 slice')
# <...>
plt.ylabel('voxels in shifted vol1 slice')
# <...>
# Correlation coefficient between them will be nearer 0
print(np.corrcoef(mid_vol0_as_1d, shifted_mid_vol1.ravel())[0, 1])
# 0.446286936383
