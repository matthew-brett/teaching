# - tell numpy to print numbers to 4 decimal places only
np.set_printoptions(precision=4, suppress=True)
# - function to print non-numpy scalars to 4 decimal places
def to_4dp(f):
    return '{0:.4f}'.format(f)
