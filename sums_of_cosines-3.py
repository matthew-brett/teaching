# When sin(d / 2) ~ 0
print('cos : sin(d/2) ~ 0;',
      predicted_cos_sum(4, np.pi * 2, 17),
      actual_cos_sum(4, np.pi * 2, 17))
# cos : sin(d/2) ~ 0; -11.1119415547 -11.1119415547
print('sin : sin(d/2) ~ 0;',
      predicted_sin_sum(4, np.pi * 2, 17),
      actual_sin_sum(4, np.pi * 2, 17))
# sin : sin(d/2) ~ 0; -12.8656424202 -12.8656424202
