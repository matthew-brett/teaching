below = mixed_p_values < (q * i / N) # True where p(i)<qi/N
max_below = np.max(np.where(below)[0]) # Max Python array index where p(i)<qi/N
print('p_i:', mixed_p_values[max_below])
# p_i: 0.00323007466783
print('i:', max_below + 1) # Python indices 0-based, we want 1-based
# i: 9
