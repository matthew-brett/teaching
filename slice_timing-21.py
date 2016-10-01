# We use the interpolator to get the values for slice 0 times:

interped_vals = lin_interper(times_slice_0)

plt.plot(times_slice_0[:10], time_course_slice_0[:10], 'b:+')
# [...]
plt.plot(times_slice_1[:10], time_course_slice_1[:10], 'r:+')
# [...]
plt.plot(times_slice_0[:10], interped_vals[:10], 'kx')
# [...]
plt.title('Using the scipy interpolation object')
# <...>
