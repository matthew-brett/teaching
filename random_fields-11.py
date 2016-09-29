# expected EC at various Z thresholds, for two dimensions
Z = np.linspace(0, 5, 1000)

def expected_ec_2d(z, resel_count):
    # From Worsley 1992
    z = np.asarray(z)
    return (resel_count * (4 * np.log(2)) * ((2*np.pi)**(-3./2)) * z) * np.exp((z ** 2)*(-0.5))
# ...
expEC = expected_ec_2d(Z, resels)
plt.plot(Z, expEC)
# [...]
plt.xlabel('Z score threshold')
# <...>
plt.ylabel('Expected EC for thresholded image')
# <...>
plt.title('Expected EC for smoothed image with %s resels' % resels)
# <...>
