# Array that is True if T1 signal >= 20, <= 30, False otherwise
t1_20_30 = (t1_slice >= 20) & (t1_slice <= 30)
# Show T1 slice, mask for T1 between 20 and 30, T2 slice
fig, axes = plt.subplots(1, 3, figsize=(8, 3))
axes[0].imshow(t1_slice)
# <...>
axes[0].set_title('T1 slice')
# <...>
axes[1].imshow(t1_20_30)
# <...>
axes[1].set_title('20<=T1<=30')
# <...>
axes[2].imshow(t2_slice)
# <...>
axes[2].set_title('T2 slice')
# <...>
