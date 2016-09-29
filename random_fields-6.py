# smooth random number image
import scipy.ndimage as spn
sd = s_fwhm / np.sqrt(8.*np.log(2)) # sigma for this FWHM
stest_img = spn.filters.gaussian_filter(test_img, sd, mode='wrap')
# >>>
def gauss_2d_varscale(sigma):
    """ Variance scaling for smoothing with 2D Gaussian of sigma `sigma`
# ...
    The code in this function isn't important for understanding
    the rest of the tutorial.
    """
    # Make a single 2D Gaussian using given sigma
    limit = sigma * 5 # go to limits where Gaussian will be at or near 0
    x_inds = np.arange(-limit, limit+1)
    y_inds = x_inds # Symmetrical Gaussian (sd same in X and Y)
    [x,y] = np.meshgrid(y_inds, x_inds)
    # http://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    gf    = np.exp(-(x*x + y*y) / (2 * sigma ** 2))
    gf    = gf/np.sum(gf)
    # Expectation of variance for this kernel
    AG    = np.fft.fft2(gf)
    Pag   = AG * np.conj(AG) # Power of the noise
    COV   = np.real(np.fft.ifft2(Pag))
    return COV[0, 0]
# ...
# Restore smoothed image to unit variance
svar = gauss_2d_varscale(sd)
scf = np.sqrt(1 / svar)
stest_img = stest_img * scf
# >>>
# display smoothed image
plt.imshow(stest_img)
# <...>
plt.set_cmap('bone')
plt.xlabel('Pixel position in X')
# <...>
plt.ylabel('Pixel position in Y')
# <...>
plt.title('Image 1 - smoothed with Gaussian kernel of FWHM %s by %s pixels' %
          (s_fwhm, s_fwhm))
# <...>
