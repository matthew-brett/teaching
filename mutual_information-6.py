fig, axes = plt.subplots(1, 2)
axes[0].hist(t1_slice.ravel(), bins=20)
# (...)
axes[0].set_title('T1 slice histogram')
# <...>
axes[1].hist(t2_slice.ravel(), bins=20)
# (...)
axes[1].set_title('T2 slice histogram')
# <...>
