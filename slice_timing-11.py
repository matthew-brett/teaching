fig, axes = plt.subplots(1, 2)
axes[0].imshow(vol0[:, :, 0])  # doctest: +SKIP
axes[0].set_title('Vol 0, z slice 0')  # doctest: +SKIP
axes[1].imshow(vol0[:, :, 1])  # doctest: +SKIP
axes[1].set_title('Vol 0, z slice 1')  # doctest: +SKIP
axes[0].autoscale(False)
axes[0].plot(vox_y, vox_x, 'rs')  # doctest: +SKIP
axes[1].autoscale(False)
axes[1].plot(vox_y, vox_x, 'rs')  # doctest: +SKIP