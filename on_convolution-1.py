import numpy as np
import matplotlib.pyplot as plt

times = np.arange(0, 40, 0.1)
n_time_points = len(times)
neural_signal = np.zeros(n_time_points)
neural_signal[(times >= 4) & (times < 9)] = 1
plt.plot(times, neural_signal)
# [...]
plt.xlabel('time (seconds)')
# <...>
plt.ylabel('neural signal')
# <...>
plt.ylim(0, 1.2)
# (...)
plt.title("Neural model for 5 second event starting at time 4")
# <...>
