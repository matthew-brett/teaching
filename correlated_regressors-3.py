# Gamma distribution from scipy
from scipy.stats import gamma

def spm_hrf(times):
    """ Return values for SPM-like HRF at given times """
    # Make output vector
    values = np.zeros(len(times))
    # Only evaluate gamma above 0 (undefined at <= 0)
    valid_times = times[times > 0]
    # Gamma pdf for the peak
    peak_values = gamma.pdf(valid_times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(valid_times, 12)
    # Combine them, put back into values vector
    values[times > 0] = peak_values - 0.35 * undershoot_values
    # Scale area under curve to 1
    return values / np.sum(values)
