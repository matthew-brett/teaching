# We plot the interpolated time course against the slice 0 times:

plt.plot(times_slice_0, interped_vals, 'r:+',
    label='interpolated slice 1 time course')
plt.plot(times_slice_0, time_course_slice_0, 'b:+',
    label='slice 0 time course')
plt.legend()
plt.title('Slice 1 time course interpolated to slice 0 times')
plt.xlabel('time (seconds)')  # doctest: +SKIP
