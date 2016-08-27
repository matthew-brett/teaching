fig, axes = plt.subplots(1, 2)
axes[0].imshow(vol0[:, :, 17])
# <...>
axes[0].set_title('vol0')
# <...>
axes[1].imshow(unshifted_vol1[:, :, 17])
# <...>
axes[1].set_title('unshifted vol1')
# <...>
