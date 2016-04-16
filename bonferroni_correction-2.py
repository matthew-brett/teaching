def sidak_thresh(alpha_fwe, n):
    return 1 - (1 - alpha_fwe)**(1./n)
# ...
print(sidak_thresh(0.05, 10))
# 0.00511619689182
