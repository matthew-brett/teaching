hrf1 = spm_hrf(times)
hrf2 = spm_hrf(times - 2) # An HRF with 2 seconds (one TR) delay
hrf1 = (hrf1 - hrf1.mean()) # Rescale and mean center
hrf2 = (hrf2 - hrf2.mean())
plt.plot(times, hrf1)
# [...]
plt.plot(times, hrf2)
# [...]
