plt.plot(times_slice_0[:10], time_course_slice_0[:10], 'b:+')
# [...]
plt.plot(times_slice_1[:10], time_course_slice_1[:10], 'r:+')
# [...]
plt.title('First 10 values for slice 0, slice 1')
# <...>
plt.xlabel('time (seconds)')
# <...>
min_y, max_y = plt.ylim()
for i in range(1, 10):
    t = times_slice_0[i]
    plt.plot([t, t], [min_y, max_y], 'k:')
    x = t
    x0 = times_slice_1[i-1]
    x1 = times_slice_1[i]
    y0 = time_course_slice_1[i-1]
    y1 = time_course_slice_1[i]
    # Apply the linear interpolation formula
    y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    plt.plot(x, y, 'kx')
# [...]
