import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats
import statsmodels.api as sm
from scipy.stats import weibull_min
import matplotlib.patches as patches

##### load the data into a dataframe
s_ds_original = pd.read_csv('stars_dataset_new.csv')
s_ds = s_ds_original[['f1_freq', 'f1_amp', 'f1_fase', 'f2_freq', 'f2_amp', 'arm_first', 'arm_sec', 'comb_suma', 'comb_rest']].copy()
s_ds['f1_amp'] = np.sqrt(s_ds['f1_amp'])
s_ds['f2_amp'] = np.sqrt(s_ds['f2_amp'])

distributions = [scipy.stats.norm, weibull_min, scipy.stats.pareto, scipy.stats.uniform, scipy.stats.expon, scipy.stats.lognorm, scipy.stats.gamma, scipy.stats.beta, scipy.stats.t]
distribution_names = ['Normal', 'Weibull', 'Pareto', 'Uniform', 'Exponential', 'LogNorm', 'Gamma', 'Beta'] #'t' 'Bernoulli','Poisson', 'Binom', Weibull

best_distribution = None
best_log_likelihood = -np.inf  # initialize with negative infinity

for feature in s_ds:
    best_distribution = None
    best_log_likelihood = -np.inf  # initialize with negative infinity
    # Set up a subplot grid
    fig, axs = plt.subplots(2, 4, figsize=(10, 10))
    fig.suptitle(feature)

    for distribution, name, ax in zip(distributions, distribution_names, axs.flatten()):
        try:
            if name == 'Weibull':
                # Fit the Weibull distribution to the data
                shape, loc, scale = weibull_min.fit(s_ds[feature], loc=0)
                params = [shape, loc, scale]
                # Compute the log-likelihood
                log_likelihood = np.sum(weibull_min.logpdf(s_ds[feature], shape, loc=loc, scale=scale))

            else:
                # Estimate the parameters of the distribution
                params = distribution.fit(s_ds[feature])

                # Compute log-likelihood
                log_likelihood = np.sum(distribution.logpdf(s_ds[feature], *params)) #higher better fit

            # print(f'{name} log-likelihood: {log_likelihood}')

            # Compute AIC and BIC
            k = len(params)  # number of parameters estimated by fit
            n = len(s_ds[feature])  # number of data points
            aic = 2*k - 2*log_likelihood #lower better fit
            bic = np.log(n)*k - 2*log_likelihood #lower better fit

            # Plot histogram of data and PDF of fitted distribution
            ax.hist(s_ds[feature], bins=30, density=True, alpha=0.6, color='g')
            # Determine range of x-values for plot
            x_start = distribution.ppf(0.001, *params)
            x_end = distribution.ppf(0.999, *params)
            x = np.linspace(x_start, x_end, 100)
            
            p = distribution.pdf(x, *params)
            ax.plot(x, p, 'k', linewidth=2)

            # Print log-likelihood, AIC, and BIC on the plot
            textstr = '\n'.join((
                f'Log-likelihood: {log_likelihood:.2f}',
                f'AIC: {aic:.2f}',
                f'BIC: {bic:.2f}'))
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top')

            ax.set_title(name)

            # Compare with the best log-likelihood so far
            if log_likelihood > best_log_likelihood:
                if p[0] < s_ds[feature][0]*10:
                    best_distribution = name
                    best_log_likelihood = log_likelihood



        except Exception as e:
            print(f'Could not fit {name} distribution: {e}')
    if best_distribution == None:
        print('No distribution was satisfactory')
    else:
        subplot_idx = distribution_names.index(best_distribution)
        if subplot_idx > 3:
            subplot_index = (1, subplot_idx-4)
        else:
            subplot_index = (0, subplot_idx)
        # Get the axis object for the selected subplot


        selected_ax = axs[subplot_index[0], subplot_index[1]]
        # selected_ax.set_title(best_distribution, color='r')

    print(f'Best distribution: {best_distribution}')
    # print(f'AIC: {aic}, BIC: {bic}')

plt.tight_layout()
plt.show()

print('Cris, now go and give a kiss to Daniele')