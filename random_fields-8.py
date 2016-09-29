def show_threshed(img, th):
    thimg = (img > th)
    plt.figure()
    plt.imshow(thimg)
    plt.set_cmap('bone')
    plt.xlabel('Pixel position in X')
    plt.ylabel('Pixel position in Y')
    plt.title('Smoothed image thresholded at Z > %s' % th)
