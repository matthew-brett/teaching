from matplotlib.ticker import MaxNLocator
ax = plt.subplot(111)
ax.plot(t1_slice.ravel(), t2_slice.ravel(), '.')
# [...]
ax.set_xlabel('T1 signal')
# <...>
ax.set_ylabel('T2 signal')
# <...>
ax.set_title('T1 vs T2 signal split into squares')
# <...>
ax.xaxis.set_minor_locator(MaxNLocator(nbins=20))
ax.yaxis.set_minor_locator(MaxNLocator(nbins=20))
ax.xaxis.grid(True, 'minor')
ax.yaxis.grid(True, 'minor')
