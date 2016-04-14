plt.plot(times_slice_0[:10], time_course_slice_0[:10], 'b:+',
    label='slice 0 time course')
# [...]
plt.plot(times_slice_1[:10], time_course_slice_1[:10], 'r:+',
    label='slice 1 time course')
# [...]
plt.legend()
# <...>
plt.title('First 10 values for slice 0, slice 1')
# <...>
plt.xlabel('time (seconds)')  # doctest: +SKIP
# <...>
