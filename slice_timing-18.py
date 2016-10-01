x1, y1, x2, y2 = 5, 6, 7, 7.5
dx, dy = x2 - x1, y2 - y1
x = x1 + dx * 0.6
y = y1 + (x-x1) * dy / dx
# Make subplots for diagram and text
fig, d_ax = plt.subplots(1, 1, figsize=(10, 4))
d_ax.plot([x1, x2], [y1, y2], 'o-')
d_ax.annotate('$(x_1, y_1)$', (x1-0.2, y1-0.3), fontsize=16)
d_ax.annotate('$(x_2, y_2)$', (x2, y2+0.2), fontsize=16)
d_ax.annotate(
    '', xy=(x1, y1), xycoords='data',
    xytext=(x2, y1), textcoords='data',
    arrowprops={'arrowstyle': '<->', 'color': 'r'})
d_ax.annotate(
    '', xy=(x2, y1), xycoords='data',
    xytext=(x2, y2), textcoords='data',
    arrowprops={'arrowstyle': '<->', 'color': 'k'})
d_ax.annotate(
    '', xy=(x1, y1+0.1), xycoords='data',
    xytext=(x, y1+0.1), textcoords='data',
    arrowprops={'arrowstyle': '<->', 'color': 'g'})
d_ax.annotate('$x_2-x_1$', (x1 + dx / 2 + 0.4, y1-0.2), fontsize=16)
d_ax.annotate('$y_2-y_1$', (x2 + 0.1, y1 + dy / 2), fontsize=16)
d_ax.annotate('$x-x1$', (x1 + 0.6, y1 + 0.2), fontsize=16)
d_ax.annotate('$x$', (x + 0.1, y1 - 1), fontsize=16)
d_ax.annotate('$y$', (x1, y + 0.1), fontsize=16)
d_ax.axis((4.3, 7.3, 4, 8))
lx, hx, ly, hy = d_ax.axis()
d_ax.plot([x, x], [ly, y], 'k:')  # line in x
d_ax.plot([lx, x], [y, y], 'k:')  # line in y
# d_ax.axis('off')
d_ax.annotate(r'slope : $\frac{y_2-y_1}{x_2-x_1}$', (4.5, y1-1.2), fontsize=20)
d_ax.annotate(r'$y = y1 + (x-x_1)\frac{y_2-y_1}{x_2-x_1}$', (4.5, y2),
              fontsize=20)
plt.setp(d_ax.get_yticklabels(), visible=False)
d_ax.yaxis.set_tick_params(size=0)
plt.setp(d_ax.get_xticklabels(), visible=False)
d_ax.xaxis.set_tick_params(size=0)
# Hide the right and top spines
d_ax.spines['right'].set_visible(False)
d_ax.spines['top'].set_visible(False)