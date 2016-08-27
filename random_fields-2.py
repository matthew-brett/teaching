import scipy.stats
normal_distribution = scipy.stats.norm()
# The inverse normal CDF
inv_n_cdf = normal_distribution.ppf
inv_n_cdf([0.95, 0.99])
# array([ 1.6449,  2.3263])
