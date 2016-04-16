from matplotlib.patches import Rectangle
fig = plt.figure(figsize=(10, 8))
ax0 = plt.gca()
ax0.axis('equal')
n_z = 16
ax0.axis((-6, n_z * 2 + 6, 0, n_z + 2))
times = np.argsort(range(0, n_z, 2) + range(1, n_z,2))
colors = np.linspace(0.5, 1, n_z)
for pos, time in enumerate(times):
    color = [colors[time]] * 3
    rect = Rectangle((0, pos), n_z * 2, 1, facecolor=color)
    ax0.add_patch(rect)
    ax0.annotate(time, (n_z , pos + 0.2))
    ax0.annotate(pos, (-2, pos + 0.2))
ax0.axis('off')
ax0.annotate('Position', (-3, n_z+1))
ax0.annotate('Acquisition order', (n_z * 0.8, n_z + 1))