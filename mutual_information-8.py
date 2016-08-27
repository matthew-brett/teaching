# Array that is True if T1 signal >= 20, <= 30, False otherwise
t1_20_30 = (t1_slice >= 20) & (t1_slice <= 30)
# Get T1 signal for voxels in given T1 signal range, 0 otherwise
t1_in_20_30 = np.where(t1_20_30, t1_slice, 0)
# Get T2 signal for voxels in given T1 signal range, 0 otherwise
t2_in_20_30 = np.where(t1_20_30, t2_slice, 0)
plt.imshow(np.hstack((t1_in_20_30, t2_in_20_30)))
# <...>
plt.title('T1, T2 signal for T1 >=20 <= 30')
# <...>
