fig, axes = plt.subplots(5, 1)
for row_no in range(5):
    axes[row_no].plot(shifted_hrfs[row_no, :])
# [...]
