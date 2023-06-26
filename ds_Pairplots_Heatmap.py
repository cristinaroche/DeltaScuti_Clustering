# Pairplot generation and heatmap

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
plt.style.use('ggplot')
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)
pd.set_option('display.expand_frame_repr', True)

Show_figures = 1
Save_figures = 0
overwrite_val = 0
other_plots = 0
pairplot = 1
heatmap = 1

if Save_figures:
    Show_figures = False



##### load the data into a dataframe
s_ds = pd.read_csv('stars_dataset_new.csv')

# Change type from 1 and 2 to HADS and LADS to appear correctly in the legend
for i in range(len(s_ds)):
    if s_ds['type'][i] == 1:
        s_ds['type'][i] = 'HADS'
    elif s_ds['type'][i] == 2:
        s_ds['type'][i] = 'LADS'

s_ds.shape
s_ds.head(20)
s_ds.columns # every column is a series
s_ds.dtypes
print(s_ds.describe()) #shows basic statistics


# Cleaning of data

# Two ways of choosing the columns
# Use .copy() at the end so python understands is a new dataframe and not a reference to the previous one
#s_ds = s_ds[['true_name', 'type', 'f1_freq', 'f1_amp', 'f1_fase', 'f2_freq', 'f2_amp', 'arm_first', 'arm_sec', 'comb_suma', 'comb_rest']].copy()
#s_ds = s_ds.drop(['Unnamed: 0'], axis=1).copy() #drop that column, axis=1 to drop the column and not the row

# Rename first column
s_ds.rename(columns={'Unnamed: 0' : 'Star_index'})

# Check nan values
s_ds.isna().sum()
# Check duplicated 
s_ds.duplicated().sum()
s_ds.loc[s_ds.duplicated(subset=['comb_rest'])] # show those cases
# Show cases data that fulfill that condition
s_ds.query('comb_suma == 1')
s_ds.query('comb_suma >= 1')
s_ds.query('comb_suma != 0')
s_ds.query('type==1').groupby('arm_first')['f1_amp'].agg(['mean','count'])
s_ds.query('type==1').groupby('arm_first')['f1_amp'].agg(['mean','count']).query('count>=2').sort_values('mean')

if other_plots:
    fig1 = plt.figure()
    ax1 = s_ds.query('type==1').groupby('arm_first')['f1_amp'].agg(['mean','count']).query('count>=2').sort_values('mean').plot(kind='barh',title='Average f1 amp by number of armonics of the fundamental')
    ax1.set_xlabel('average f1 amp')

    ##### Individual feature exploratory analysis
    ### Histogram
    # Counts how many cases in each value happen (histogram), shows from more to less cases
    s_ds['comb_rest'].value_counts() # first column data value, second number of cases
    s_ds['comb_rest'].value_counts().head(10) # 10 most common values
    fig2 = plt.figure()
    ax2 = s_ds['comb_rest'].value_counts().head(10).plot(kind='bar', title='Top common combination of rest')
    ax2.set_xlabel('Number of rest combinations')
    ax2.set_ylabel('Count')

    # ax = s_ds['comb_rest'].plot(kind='hist',
    #                           bins=20,
    #                           title='Histogram: Combination of rest')
    # ax.set_xlabel('Number of rest combinations')

    ### Density plot: similar to histogram but normalized and without bins
    # ax = s_ds['comb_rest'].plot(kind='kde',
    #                           title='Density plot: Combination of rest')
    # ax.set_xlabel('Number of rest combinations')

    ##### Multiple feature exploratory analysis
    # 2 or 3 feature comparison: Scatter plots
    # Pandas scatter plot: Compare two features side by side
    # ax = s_ds.plot(kind='scatter', x='type', y='f1_amp')
    # ax = s_ds.plot(kind='scatter', x='type', y='arm_first')
    # ax = s_ds.plot(kind='scatter', x='f1_amp', y='arm_first')
    # ax = s_ds.plot(kind='scatter', x='f1_amp', y='f1_freq')
    # ax = s_ds.plot(kind='scatter', x='type', y='f2_amp')
    # ax = s_ds.plot(kind='scatter', x='type', y='arm_sec')
    # plt.show() # removes a line of info about the plot that is not important

    # Seaborn scatter plot (better)
    # sns.scatterplot(x='f1_amp', y='f1_freq', data=s_ds)
    # sns.scatterplot(x='f1_amp', y='f1_freq', hue = 'type', data=s_ds) # Puts colours per hue

    ### Plot graphs with discrete values with the count on top of the dots
    # ax = s_ds.plot(kind='scatter', x='type', y='arm_first')
    # x=s_ds['type']
    # y=s_ds['arm_first']
    # grouped = s_ds.groupby(['type','arm_first'])
    # grouped_size = s_ds.groupby(['type','arm_first']).size()
    # grouped_size.values 
    # loc = list(grouped.indices.keys())
    # for i in range(len(grouped.indices.keys())):
    #     plt.text(loc[i][0], loc[i][1], str(grouped_size.values[i]), ha='center', va='bottom') #option to add text to the plot

    # ax = s_ds.plot(kind='scatter', x='type', y='f2_amp')
    # x=s_ds['type']
    # y=s_ds['f2_amp']
    # grouped = s_ds.groupby(['type','f2_amp'])
    # grouped_size = s_ds.groupby(['type','f2_amp']).size()
    # grouped_size.values 
    # loc = list(grouped.indices.keys())
    # for i in range(len(grouped.indices.keys())):
    #     plt.text(loc[i][0], loc[i][1], str(grouped_size.values[i]), ha='center', va='bottom') #option to add text to the plot


if pairplot:
    ##### Compare more than 3 features
    fig3 = plt.figure()
    colors = ['#D55E00', '#0072B2'] # 'colorblind' palette from sns
    sns.set_palette(colors)
    s_ds = s_ds.rename(columns={'type': 'Type', 'f1_freq': 'f1', 'f1_amp_sqrt': 'A1', 'f1_fase': 'P1', 'f2_freq': 'f2', 'f2_amp_sqrt': 'A2', 'arm_first': 'Harm1', 'arm_sec': 'Harm2', 'comb_suma': 'AddComb', 'comb_rest': 'SubComb', 'Cluster': 'Cluster'})
    pair = sns.pairplot(s_ds, hue='Type', diag_kind= 'hist', vars=['f1', 'A1', 'P1', 'f2', 'A2', 'Harm1', 'Harm2', 'AddComb', 'SubComb'], plot_kws={"s": 20}, diag_kws = {'alpha':0.55, 'bins':20},  palette = colors) # kind='reg' #size of points: plot_kws={"s": 20}
    #sns.pairplot(kind = 'scatter')

if heatmap:
    ### Show correlation between features
    s_ds_corr = s_ds[['f1', 'A1', 'P1', 'f2', 'A2', 'Harm1', 'Harm2', 'AddComb', 'SubComb']].corr()
    ### Heatmap
    fig4 = plt.figure()
    heat = sns.heatmap(s_ds_corr, annot=True)

if Show_figures:    
    plt.show()

if Save_figures:
    # save the figure in png format
    if other_plots:
        fig1.savefig('./new/Figures_new/Exploratory_Stats_Analysis/nose1.png', overwrite=overwrite_val)
        fig2.savefig('./new/Figures_new/Exploratory_Stats_Analysis/nose2.png', overwrite=overwrite_val)

        # save the figure in pdf format
        fig1.savefig('./new/Figures_new/Exploratory_Stats_Analysis/nose1.pdf', overwrite=overwrite_val)
        fig2.savefig('./new/Figures_new/Exploratory_Stats_Analysis/nose2.pdf', overwrite=overwrite_val)

    if pairplot:
        fig3.savefig('./new/Figures_new/Exploratory_Stats_Analysis/Pairplot.png', overwrite=overwrite_val)
        # save the figure in pdf format
        fig3.savefig('./new/Figures_new/Exploratory_Stats_Analysis/Pairplot.pdf', overwrite=overwrite_val)

    if heatmap:
        fig4.savefig('./new/Figures_new/Exploratory_Stats_Analysis/Heatmap.png', overwrite=overwrite_val)
        # save the figure in pdf format
        fig4.savefig('./new/Figures_new/Exploratory_Stats_Analysis/Heatmap.pdf', overwrite=overwrite_val)
