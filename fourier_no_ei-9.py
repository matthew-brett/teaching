full_circle = np.linspace(0, 2 * np.pi, 500)
plt.plot(np.cos(full_circle), np.sin(full_circle))
pts = np.c_[np.cos(vec_r_1), np.sin(vec_r_1)]
plt.plot(pts[:, 0], pts[:, 1], 'o')
ax = plt.gca()
for i in [0, 1, 2, 3, 30, 31]:
    pos = pts[i]
    ax.annotate('n=%d' % i, xy=pos, xytext=pos + [0.1, 0])
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('center')
ax.spines['top'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)
plt.axis('equal')
plt.title("Angle ($r^1$) vector values as positions on a circle")