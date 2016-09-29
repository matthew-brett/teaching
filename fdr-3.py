import scipy.stats as sst
normal_distribution = sst.norm(loc=0,scale=1.) #loc is the mean, scale is the variance.
# The normal CDF
p_values = normal_distribution.cdf(z_values)
