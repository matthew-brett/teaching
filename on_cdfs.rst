###############################################
p values from cumulative distribution functions
###############################################

.. nbplot::

    >>> import numpy as np
    >>> np.set_printoptions(precision=4)  # print arrays to 4 decimal places
    >>> import matplotlib.pyplot as plt

Imagine I have a t statistic with 20 degrees of freedom.

`Scipy`_ provides a *t distribution object* that we can use to get values from
the t statistic `probability density function`_ (PDF).

As a start, we plot the PDF for a t statistic with 20 degrees of freedom:

.. nbplot::

    >>> import scipy.stats
    >>> # Make a t distribution object for t with 20 degrees of freedom
    >>> t_dist = scipy.stats.t(20)
    >>> # Plot the PDF
    >>> t_values = np.linspace(-4, 4, 1000)
    >>> plt.plot(t_values, t_dist.pdf(t_values))
    [...]
    >>> plt.xlabel('t value')
    <...>
    >>> plt.ylabel('probability for t value')
    <...>
    >>> plt.title('PDF for t distribution with df=20')
    <...>

The t distribution object ``t_dist`` can also give us the `cumulative
distribution function`_ (CDF).  The CDF gives the area under the curve of the
PDF at and to the left of the given t value:

.. nbplot::

    >>> # Plot the CDF
    >>> plt.plot(t_values, t_dist.cdf(t_values))
    [...]
    >>> plt.xlabel('t value')
    <...>
    >>> plt.ylabel('probability for t value <= t')
    <...>
    >>> plt.title('CDF for t distribution with df=20')
    <...>

Say I have a t value $x$ drawn from a t distribution with 20 degrees of
freedom.  The PDF gives the probability for given values of $x$.  Because it
is a probability density, the sum of the probabilities of all possible
outcomes (values for $x$) of $-\infty < x < \infty$ must be 1.  Therefore the
total area under the PDF curve is 1, and the maximum value of the CDF is 1.

The CDF gives us the area under the PDF curve at and to the left of a given t
value $x$.  Therefore it is the probability that we will observe a value $x <=
t$ if we sample a value $x$ from a t distribution of (here) 20 degrees of
freedom.

.. nbplot::

    # Show relationship of PDF to CDF for three threshold values.
    thresholds = (-1.5, 0, 1.5)
    pdf_values = t_dist.pdf(t_values)
    cdf_values = t_dist.cdf(t_values)
    fill_color = (0, 0, 0, 0.1)
    line_color = (0, 0, 0, 0.5)
    fig, ax = plt.subplots(2, len(thresholds), figsize=(10, 6))
    for i, t in enumerate(thresholds):
        ax[0, i].plot(t_values, cdf_values)
        ax[1, i].plot(t_values, pdf_values)
        # Fill area at and to the left of threshold t.
        ax[1, i].fill_between(t_values, pdf_values,
                              where=t_values <= t,
                              color=fill_color)
        p = t_dist.pdf(t)  # Probability density at this threshold.
        # Line showing position of t on x-axis of PDF plot.
        ax[1, i].plot([t, t],
                      [0, p], color=line_color)
        c = t_dist.cdf(t)  # Cumulative distribution at this threshold.
        # Lines showing t and CDF value for t on CDF plot.
        ax[0, i].plot([t, t, ax[0, i].axis()[0]],
                      [0, c, c], color=line_color)
        ax[0, i].set_title('t = {:.1f}, area = {:.2f}'.format(t, c))


For example, say I have drawn a t value $x$ at random from a t distribution
with 20 degrees of freedom.  The probability that $\mathbb{P}(x <=1.5)$ is:

.. nbplot::

    >>> # Area of PDF at and to the left of 1.5
    >>> t_dist.cdf(1.5)
    0.9253...

The total area under the PDF is 1, and the maximum value for the CDF is 1.
Therefore the area of the PDF to the *right* of 1.5 must be:

.. nbplot::

    >>> # Area of PDF to the right of 1.5
    >>> 1 - t_dist.cdf(1.5)
    0.0746...

This is the probability that our t value $x$ will be $> 1.5$.  In general,
when we sample a value $x$ at random from a t distribution with $d$ degrees of
freedom, the probability that $x > q$ is given by:

.. math::

    \mathbb{P}(x > q) = 1 - \mathrm{CDF}_d(q)

where $\mathrm{CDF}_d$ is the cumulative distribution function for a t value
with $d$ degrees of freedom.

.. include:: links_names.inc
