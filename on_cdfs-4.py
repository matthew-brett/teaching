# Show relationship of PDF to CDF
thresholds = (-1.5, 0, 1.5)
pdf_values = t_dist.pdf(t_values)
cdf_values = t_dist.cdf(t_values)
fill_color = (0, 0, 0, 0.1)
line_color = (0, 0, 0, 0.5)
fig, ax = plt.subplots(2, len(thresholds), figsize=(10, 6))
for i, t in enumerate(thresholds):
    ax[0, i].plot(t_values, cdf_values)
    ax[1, i].plot(t_values, pdf_values)
    ax[1, i].fill_between(t_values, pdf_values,
                          where=t_values <= t,
                          color=fill_color)
    p = t_dist.pdf(t)  # Probability density at this threshold
    ax[1, i].plot([t, t],
                  [0, p], color=line_color)
    c = t_dist.cdf(t)  # Cumulative density at this threshold
    ax[0, i].plot([t, t, ax[0, i].axis()[0]],
                  [0, c, c], color=line_color)
    ax[0, i].set_title('t = {:.1f}, area = {:.2f}'.format(t, c))