plt.plot(times_slice_0[:10], time_course_slice_0[:10], 'b:+')
# [...]
plt.plot(times_slice_1[:10], time_course_slice_1[:10], 'r:+')
# [...]
plt.title('First 10 values for slice 0, slice 1')
# <...>
plt.xlabel('time (seconds)')  # doctest: +SKIP
# <...>
min_y, max_y = plt.ylim()
for i in range(1, 10):
    t = times_slice_0[i]
    plt.plot([t, t], [min_y, max_y], 'k:')
# [...]
