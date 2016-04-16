signal = hrf1 + hrf2
plt.plot(hrf1, label='hrf1')
# [...]
plt.plot(hrf2, label='hrf2')
# [...]
plt.plot(signal, label='signal (combined hrfs)')
# [...]
plt.legend()
# <...>
