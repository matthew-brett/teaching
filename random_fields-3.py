# Constants for image simulations etc
shape = [128, 128] # No of pixels in X, Y
n_voxels = np.prod(shape) # Total pixels
s_fwhm = 8 # Smoothing in number of pixels in x, y
seed = 1939 # Seed for random no generator
alpha = 0.05 # Default alpha level
# >>>
# Image of independent random nos
np.random.seed(seed) # Seed the generator to get same numbers each time
test_img = np.random.standard_normal(shape)
plt.imshow(test_img)
# <...>
plt.set_cmap('bone')
plt.xlabel('Pixel position in X')
# <...>
plt.ylabel('Pixel position in Y')
# <...>
plt.title('Image 1 - array of independent random numbers')
# <...>
