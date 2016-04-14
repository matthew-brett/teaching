from scipy.interpolate import InterpolatedUnivariateSpline as Interp

# This class can do more fancy interpolation, but we will use it for linear
# interpolation (``k=1`` argument below):

lin_interper = Interp(times_slice_1, time_course_slice_1, k=1)
type(lin_interper)
# ...InterpolatedUnivariateSpline
