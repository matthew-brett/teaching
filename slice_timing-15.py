plt.plot(times_slice_0, time_course_slice_0, 'b:+',
    label='slice 0 time course')
# [...]
plt.plot(times_slice_1, time_course_slice_1, 'r:+',
    label='slice 1 time course')
# [...]
plt.legend()
# <...>
plt.title('Time courses for slice 0, slice 1')
# <...>
plt.xlabel('time (seconds)')  # doctest: +SKIP
# <...>
