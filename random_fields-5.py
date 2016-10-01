# Divide into FWHM chunks and fill square from mean value
sqmean_img = test_img.copy()
for i in range(0, shape[0], s_fwhm):
    i_slice = slice(i, i+s_fwhm)
    for j in range(0, shape[1], s_fwhm):
        j_slice = slice(j, j+s_fwhm)
        vals = sqmean_img[i_slice, j_slice]
        sqmean_img[i_slice, j_slice] = vals.mean()
# Multiply up to unit variance again
sqmean_img *= s_fwhm
# Show as image
plt.imshow(sqmean_img)
# <...>
plt.set_cmap('bone')
plt.xlabel('Pixel position in X')
# <...>
plt.ylabel('Pixel position in Y')
# <...>
plt.title('Taking means over %s by %s elements from image 1' % (s_fwhm, s_fwhm))
# <...>
