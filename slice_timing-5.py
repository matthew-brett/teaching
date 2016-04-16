# Configure matplotlib

plt.rcParams['image.cmap'] = 'gray'  # default gray colormap
plt.rcParams['image.interpolation'] = 'nearest'

# Show sagittal section:

plt.imshow(vol0[31, :, :].T, origin='bottom left')  # doctest: +SKIP
# <...>
plt.title('Sagittal section through first volume') # doctest: +SKIP
# <...>
plt.xlabel('x axis')  # doctest: +SKIP
# <...>
plt.ylabel('z axis')  # doctest: +SKIP
# <...>
