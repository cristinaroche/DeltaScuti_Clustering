# Plots histograms and boxplots for the stars dataframe 
# Several option can be selected so it runs faster for visualization

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from PIL import Image
import os
import pickle

plt.style.use('ggplot')
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)
pd.set_option('display.expand_frame_repr', True)

Show_figures = 1
Save_figures = 0
Save_outliers = 0
overwrite_val = 0
histograms = 1
boxplots = 1
boxplots_exp = 0

# if box_plots_exp need to be printed, also boxplots option has to be on
if boxplots_exp == 1:
    boxplots = 1

if Save_figures or Save_outliers:
    Show_figures = False



##### load the data into a dataframe
s_ds_original = pd.read_csv('stars_dataset_new.csv')
s_ds = s_ds_original[['f1_freq', 'f1_amp', 'f1_fase', 'f2_freq', 'f2_amp', 'arm_first', 'arm_sec', 'comb_suma', 'comb_rest']].copy()
s_ds2 = s_ds_original[['type','f1_freq', 'f1_amp_sqrt', 'f1_fase', 'f2_freq', 'f2_amp_sqrt', 'arm_first', 'arm_sec', 'comb_suma', 'comb_rest']].copy()

# Change type from 1 and 2 to HADS and LADS to appear correctly in the legend
for i in range(len(s_ds_original)):
    if s_ds_original['type'][i] == 1:
        s_ds_original['type'][i] = 'HADS'
    elif s_ds_original['type'][i] == 2:
        s_ds_original['type'][i] = 'LADS'

variables = s_ds.columns
titles = [r'Frequency fundamental mode $[Hz]$', r'Amplitude fundamental mode $[e^-/s]$', r'Phase fundamental mode $[deg]$', r'Frequency first overtone $[Hz]$', r'Amplitude first overtone  $[e^{-1}/s]$', 
          'Harmonics fundamental mode', 'Harmonics first overtone', 'Number of additive combinations', 'Number of subtractive combinations']

# Transform from amplitude in power of two due to spectrum to amplitude (e^-1/s)
s_ds['f1_amp'] = np.sqrt(s_ds['f1_amp']).copy()
s_ds['f2_amp'] = np.sqrt(s_ds['f2_amp']).copy()
if histograms:
    ##### ALL histogram: 
    # Create a 3x3 grid of plots
    fig1, axs1 = plt.subplots(3, 3, figsize=(12, 10))
    fig1.suptitle('Histograms in percentage', fontsize=15)
    # Flatten the axes array to simplify indexing
    axs1 = axs1.ravel()
    # Loop over the data and labels and plot each variable
    for i, data in enumerate(variables):
        # Calculate the mean and standard deviation
        mean = np.mean(s_ds[data])
        std = np.std(s_ds[data])
        
        # Plot the histogram
        #axs[i].hist(s_ds[data], bins=30, alpha=0.5, color='b')
        if  data == 'comb_suma':
            sns.histplot(data=s_ds, x=data, stat='percent', bins=30, kde=True, palette=None, color=None, legend=False, ax=axs1[i]) #data=lnp.log10(s_ds+1)
        elif data == 'f2_amp':
            sns.histplot(data=s_ds, x=data, stat='percent', bins=30, kde=True, palette=None, color=None, legend=False, ax=axs1[i])
        elif data == 'arm_first':
            sns.histplot(data=s_ds, x=data, stat='percent', binwidth=1, kde=True, palette=None, color=None, legend=False, ax=axs1[i])
        elif data == 'arm_sec':
            sns.histplot(data=s_ds, x=data, stat='percent', binwidth=1, kde=True, palette=None, color=None, legend=False, ax=axs1[i])
        else:
            sns.histplot(data=s_ds, x=data, stat='percent', bins='auto', kde=True, palette=None, color=None, legend=False, ax=axs1[i])
        # Add a title and label to the plot
        axs1[i].set_title(titles[i], fontsize=10)
        axs1[i].set_xlabel('Value')
        axs1[i].set_ylabel('Percentage [%]')
    
    # ##### ALL histogram exponential data: 
    # # Create a 3x3 grid of plots
    # fig1, axs1 = plt.subplots(3, 3, figsize=(12, 10))
    # fig1.suptitle('Histograms in percentage, log10(data+1)', fontsize=15)
    # # Flatten the axes array to simplify indexing
    # axs1 = axs1.ravel()
    # # Loop over the data and labels and plot each variable
    # for i, data in enumerate(variables):
    #     # Calculate the mean and standard deviation
    #     mean = np.mean(s_ds[data])
    #     std = np.std(s_ds[data])
    #     s_ds = np.log10(s_ds+1)
    #     # Plot the histogram
    #     #axs[i].hist(s_ds[data], bins=30, alpha=0.5, color='b')
    #     if  data == 'comb_suma':
    #         sns.histplot(data=s_ds, x=data, stat='percent', bins=30, kde=True, palette=None, color='#1f77b4', legend=False, ax=axs1[i]) #data=lnp.log10(s_ds+1)
    #     elif data == 'f2_amp':
    #         sns.histplot(data=s_ds, x=data, stat='percent', bins=30, kde=True, palette=None, color='#1f77b4', legend=False, ax=axs1[i])
    #     elif data == 'arm_first':
    #         sns.histplot(data=s_ds, x=data, stat='percent', kde=True, palette=None, color='#1f77b4', legend=False, ax=axs1[i])
    #     elif data == 'arm_sec':
    #         sns.histplot(data=s_ds, x=data, stat='percent', binwidth=1, kde=True, palette=None, color='#1f77b4', legend=False, ax=axs1[i])
    #     else:
    #         sns.histplot(data=s_ds, x=data, stat='percent', bins='auto', kde=True, palette=None, color='#1f77b4', legend=False, ax=axs1[i])

    # Adjust the spacing between the plots
    plt.subplots_adjust(wspace=0.3, hspace=0.6)

    s_ds['arm_sec'].value_counts()



    ##### f1 amplitude histogram
    fig0 = plt.figure()
    colors = ['#D55E00', '#0072B2'] # 'colorblind' palette from sns
    sns.set_palette(colors)
    fig0.suptitle(r'Fundamental mode amplitude histogram in percentage $[e^-/s]$', fontsize=15)
    axs = sns.histplot(data=s_ds, x=s_ds['f1_amp'], stat='percent', kde= True, hue = list(s_ds_original['type']), bins='auto', palette=colors, color=None, legend=True)
    # Add a title and label to the plot
    axs.set_xlabel(r'Fundamental mode amplitude $[e^-/s]$')
    axs.set_ylabel('Percentage of cases [%]')

    fig = plt.figure()
    colors = ['#D55E00', '#0072B2'] # 'colorblind' palette from sns
    sns.set_palette(colors)
    fig.suptitle(r'Fundamental mode amplitude histogram in percentage $[e^-/s]$', fontsize=15)
    axs = sns.histplot(data=s_ds, x=s_ds['f1_amp'], stat='percent', hue = list(s_ds_original['type']), bins='auto', binwidth=1, palette=colors, color=None, legend=True)
    mean = np.mean(s_ds['f1_amp'])
    std = np.std(s_ds['f1_amp'])
    median = np.median(s_ds['f1_amp'])
    q1 = np.percentile(s_ds['f1_amp'], 25)
    q3 = np.percentile(s_ds['f1_amp'], 75)
    iqr = q3 - q1
    # Add a vertical line for the mean and standard deviation
    axs.axvline(x=mean, color='k', linestyle='dashed', linewidth=1)
    axs.axvline(mean + 3*std, color='#76EEC6', linestyle='dashed', linewidth=1)
    axs.axvline(median, color='r', linewidth=0.5)
    axs.axvline(q3+1.5*iqr, color='firebrick', linewidth=0.5)
    axs.axvline(mean - std, color='#458B74', linestyle='dashed', linewidth=1)
    axs.axvline(mean + std, color='#458B74', linestyle='dashed', linewidth=1)
    axs.axvline(mean + 2*std, color='#66CDAA', linestyle='dashed', linewidth=1)
    # Add text for the mean and standard deviation
    text = r'$3\sigma = $' + f'{mean + 3*std:.2f}'
    outliers_lim = mean + 3*std
    n_outliers = len(s_ds.query('f1_amp' + '>=' + str(outliers_lim)))
    text2 = f'Outliers ' + f'{n_outliers}'
    axs.text(mean+ 3*std, 1, text+'.\n'+text2, fontsize=8, color='k')
    # Add a title and label to the plot
    axs.set_xlabel(r'Fundamental mode amplitude $[e^-/s]$')
    axs.set_ylabel('Percentage of cases [%]')


    # fig00 = plt.figure()
    # colors = ['#D55E00', '#0072B2'] # 'colorblind' palette from sns
    # sns.set_palette(colors)
    # sns.violinplot(data=s_ds, Y=s_ds['f1_amp'], stat='percent', hue = list(s_ds_original['type']), palette=colors, split=True, legend=True)

    ##### Additive and subtractive combinations histogram
    fig00 = plt.figure()
    colors = ['#D55E00', '#0072B2'] # 'colorblind' palette from sns
    sns.set_palette(colors)
    fig00.suptitle('Number of additive combinations histogram in percentage', fontsize=15)
    axs = sns.histplot(data=s_ds, x=s_ds['comb_suma'], stat='percent', kde= True, hue = list(s_ds_original['type']), bins='auto', binwidth=1, palette=colors, color=None, legend=True)
    # Add a title and label to the plot
    axs.set_xlabel('Number of additive combinations')
    axs.set_ylabel('Percentage of cases [%]')
    axs.set_xlim([0, 20])
    # Set the x-axis tick frequency
    axs.xaxis.set_major_locator(MultipleLocator(2))

    fig01 = plt.figure()
    colors = ['#D55E00', '#0072B2'] # 'colorblind' palette from sns
    sns.set_palette(colors)
    fig.suptitle('Number of subtractive combinations histogram in percentage', fontsize=15)
    axs = sns.histplot(data=s_ds, x=s_ds['comb_rest'], stat='percent', hue = list(s_ds_original['type']), bins='auto', binwidth=0.5, palette=colors, color=None, legend=True)
    axs.set_xlabel('Number of subtractive combinations')
    axs.set_ylabel('Percentage of cases [%]')
    axs.set_xlim([0, 40])

    ##### HADS LADS histogram: 
    # Create a 3x3 grid of plots
    fig2, axs2 = plt.subplots(3, 3, figsize=(12, 10))
    fig2.suptitle('HADS and LADS histograms in percentage',fontsize=15)
    # Flatten the axes array to simplify indexing
    axs2 = axs2.ravel()
    # Loop over the data and labels and plot each variable
    for i, data in enumerate(variables):
        # Calculate the mean and standard deviation
        mean = np.mean(s_ds[data])
        std = np.std(s_ds[data])
        
        colors = ['#D55E00', '#0072B2'] # 'colorblind' palette from sns
        sns.set_palette(colors)
        # Plot the histogram
        #axs[i].hist(s_ds[data], bins=30, alpha=0.5, color='b')
        #sns.histplot(data=s_ds, x=data, stat='count', bins='auto', kde=True, palette='colorblind', color=None, legend=True, ax=axs2[i])
        if i == (len(variables)-1):
            sns.histplot(data=s_ds, x=data, stat='percent', bins='auto', kde=True, hue = list(s_ds_original['type']), palette = colors,  color=None, legend=True, ax=axs2[i])
        elif data == 'comb_suma':
            sns.histplot(data=s_ds, x=data, stat='percent', bins=30, kde=False, hue = list(s_ds_original['type']), palette = colors,  color=None, legend=False, ax=axs2[i])
        elif data == 'f2_amp':
            sns.histplot(data=s_ds, x=data, stat='percent', bins=30, kde=True, hue = list(s_ds_original['type']), palette = colors,  color=None, legend=False, ax=axs2[i])
        elif data == 'arm_first':
            sns.histplot(data=s_ds, x=data, stat='percent', binwidth=1, kde=True, hue = list(s_ds_original['type']), palette = colors,  color=None, legend=False, ax=axs2[i])
        elif data == 'arm_sec':
            sns.histplot(data=s_ds, x=data, stat='percent', binwidth=1, kde=True, hue = list(s_ds_original['type']), palette = colors,  color=None, legend=False, ax=axs2[i])
        else:
            sns.histplot(data=s_ds, x=data, stat='percent', bins='auto', kde=True, hue = list(s_ds_original['type']), palette = colors, color=None, legend=False, ax=axs2[i])
        
        # For plotting the histogram with polynomials
        # if i == (len(variables)-1):
        #     sns.histplot(data=s_ds, x=data, stat='percent', bins='auto', kde=False, hue = list(s_ds_original['type']), palette = colors,  color=None, legend=True, ax=axs3[i])
        # elif data == 'comb_suma':
        #     sns.histplot(data=s_ds, x=data, stat='percent', bins=30, kde=False, hue = list(s_ds_original['type']), palette = colors,  color=None, legend=False, ax=axs3[i])
        # elif data == 'f2_amp':
        #     sns.histplot(data=s_ds, x=data, stat='percent', bins=30, kde=False, hue = list(s_ds_original['type']), palette = colors,  color=None, legend=False, ax=axs3[i])
        # else:
        #     sns.histplot(data=s_ds, x=data, stat='percent', bins='auto', kde=False, hue = list(s_ds_original['type']), palette = colors, color=None, legend=False, ax=axs3[i])
        

        # Add a title and label to the plot
        axs2[i].set_title(titles[i], fontsize=10)
        axs2[i].set_xlabel('Value')
        axs2[i].set_ylabel('Percentage [%]')

    # Adjust the spacing between the plots
    plt.subplots_adjust(wspace=0.3, hspace=0.6)







    ##### HADS LADS HISTOGRAMS WITH OUTLIERS AND DATA OF ALL OUTLIERS
    ## Initialize variables for the outliers information output
    outliers_ds = pd.DataFrame()
    # create a new empty dataframe with the same structure as df1
    outliers_ds = s_ds_original.copy()  # copy() method to copy the structure of s_ds_original
    outliers_ds = outliers_ds.iloc[0:0] # iloc[0:0] syntax to clear out any rows that may have been copied
    outliers_ds['outlier_reason'] = []  # Add new column

    # Create a 3x3 grid of plots
    fig3, axs3 = plt.subplots(3, 3, figsize=(12, 10))
    fig3.suptitle('Histograms with statistical information and outliers for '+ '$3\sigma$', fontsize=15)
    # Flatten the axes array to simplify indexing
    axs3 = axs3.ravel()
    # Loop over the data and labels and plot each variable
    for i, data in enumerate(variables):
        # Calculate the mean and standard deviation
        mean = np.mean(s_ds[data])
        std = np.std(s_ds[data])
        median = np.median(s_ds[data])
        q1 = np.percentile(s_ds[data], 25)
        q3 = np.percentile(s_ds[data], 75)
        iqr = q3 - q1
        colors = ['#D55E00', '#0072B2'] # 'colorblind' palette from sns
        sns.set_palette(colors)
        # Plot the histogram
        if i == (len(variables)-1):
            sns.histplot(data=s_ds, x=data, stat='percent', bins='auto', kde=False, hue = list(s_ds_original['type']), palette = colors,  color=None, legend=True, ax=axs3[i])
        elif data == 'comb_suma':
            sns.histplot(data=s_ds, x=data, stat='percent', bins=30, kde=False, hue = list(s_ds_original['type']), palette = colors,  color=None, legend=False, ax=axs3[i])
        elif data == 'f2_amp':
            sns.histplot(data=s_ds, x=data, stat='percent', bins=30, kde=False, hue = list(s_ds_original['type']), palette = colors,  color=None, legend=False, ax=axs3[i])
        elif data == 'arm_first':
            sns.histplot(data=s_ds, x=data, stat='percent', binwidth=1, kde=True, hue = list(s_ds_original['type']), palette = colors,  color=None, legend=False, ax=axs3[i])
        elif data == 'arm_sec':
            sns.histplot(data=s_ds, x=data, stat='percent', binwidth=1, kde=True, hue = list(s_ds_original['type']), palette = colors,  color=None, legend=False, ax=axs3[i])
        else:
            sns.histplot(data=s_ds, x=data, stat='percent', bins='auto', kde=False, hue = list(s_ds_original['type']), palette = colors, color=None, legend=False, ax=axs3[i])

        
        # Add a vertical line for the mean and standard deviation
        axs3[i].axvline(x=mean, color='k', linestyle='dashed', linewidth=1)
        axs3[i].axvline(mean + 3*std, color='#76EEC6', linestyle='dashed', linewidth=1)
        axs3[i].axvline(median, color='r', linewidth=0.5)
        axs3[i].axvline(q3+1.5*iqr, color='firebrick', linewidth=0.5)

        axs3[i].axvline(mean - std, color='#458B74', linestyle='dashed', linewidth=1)
        axs3[i].axvline(mean + std, color='#458B74', linestyle='dashed', linewidth=1)
        #axs3[i].axvline(mean - 2*std, color='#66CDAA', linestyle='dashed', linewidth=1)
        axs3[i].axvline(mean + 2*std, color='#66CDAA', linestyle='dashed', linewidth=1)
        #axs3[i].axvline(mean - 3*std, color='#76EEC6', linestyle='dashed', linewidth=1)

        # Add text for the mean and standard deviation
        text = r'$3\sigma = $' + f'{mean + 3*std:.2f}'
        outliers_lim = mean + 3*std
        n_outliers = len(s_ds.query(data + '>=' + str(outliers_lim)))
        text2 = f'Outliers ' + f'{n_outliers}'
        axs3[i].text(mean+ 3*std, 1, text+'.\n'+text2, fontsize=8, color='k')

        # ymin, ymax = axs3[i].get_ylim()
        # axs3[i].text(mean+0.2, ymax-5, f'\u03BC', fontsize=8, color='k')


        # Save outliers
        if Save_outliers:
            outliers_dict_temp = []
            outliers_dict_temp = s_ds.query(data + '>=' + str(outliers_lim))
            outliers_list = list(outliers_dict_temp.index)
            for j in range(len(outliers_list)):
                outliers_ds.loc[len(outliers_ds)] = s_ds_original.iloc[outliers_list[j]] # Loc: access a row in a dataset, logic saves data of the whole structure in next row
                outliers_ds.loc[len(outliers_ds)-1,'outlier_reason'] = data

        # Add a title and label to the plot
        axs3[i].set_title(titles[i], fontsize=10)
        axs3[i].set_xlabel('Value')
        axs3[i].set_ylabel('Percentage [%]')


    fig3.legend(['mean', '$3\sigma$', 'median', 'q3+1.5*IQR'],loc = 'lower right')
    #plt.subplots_adjust(left=0.07, right=0.93, wspace=0.25, hspace=0.35)
    # Ajustar diseño de la figura
    #fig3.tight_layout()

    # Adjust the spacing between the plots
    plt.subplots_adjust(wspace=0.3, hspace=0.6)


    # See repeated stars that are outliers and saving everything
    if Save_outliers:
        list_outliers = outliers_ds['Unnamed: 0'].value_counts()
        with open('./new/Output_new/Exploratory_Stats_Analysis/outliers_info.pkl', 'wb') as f:
            pickle.dump(outliers_ds, f)
        with open('./new/Output_new/Exploratory_Stats_Analysis/outliers_list.pkl', 'wb') as f:
            pickle.dump(list_outliers, f)
        list_outliers.to_csv('./new/Output_new/Exploratory_Stats_Analysis/Outliers_list.csv', index=True)
        outliers_ds.to_csv('./new/Output_new/Exploratory_Stats_Analysis/Outliers_info.csv', index=True)







if boxplots:
    ##### BOXPLOTS ALL WITH OUTLIERS
    ## Initialize variables for the outliers information output
    outliers_ds_IQR = pd.DataFrame()
    # create a new empty dataframe with the same structure as df1
    outliers_ds_IQR = s_ds_original.copy()  # copy() method to copy the structure of s_ds_original
    outliers_ds_IQR = outliers_ds_IQR.rename(columns={'f1_freq': 'f1', 'f1_amp_sqrt': 'A1', 'f1_fase': 'P1', 'f2_freq': 'f2', 'f2_amp_sqrt': 'A2', 'arm_first': 'Harm1', 'arm_sec': 'Harm2', 'comb_suma': 'AddComb', 'comb_rest': 'SubComb', 'Cluster': 'Cluster'})
    outliers_ds_IQR = outliers_ds_IQR.iloc[0:0] # iloc[0:0] syntax to clear out any rows that may have been copied
    outliers_ds_IQR['outlier_reason_IQR'] = []  # Add new column

    # Create a 3x3 grid of plots
    fig4, axs4 = plt.subplots(3, 3, figsize=(16, 10))


    # Flatten the axes array to simplify indexing
    axs4 = axs4.ravel()
    s_ds = s_ds_original[['f1_freq', 'f1_amp_sqrt', 'f1_fase', 'f2_freq', 'f2_amp_sqrt', 'arm_first', 'arm_sec', 'comb_suma', 'comb_rest']].copy()
    s_ds = s_ds.rename(columns={'f1_freq': 'f1', 'f1_amp_sqrt': 'A1', 'f1_fase': 'P1', 'f2_freq': 'f2', 'f2_amp_sqrt': 'A2', 'arm_first': 'Harm1', 'arm_sec': 'Harm2', 'comb_suma': 'AddComb', 'comb_rest': 'SubComb', 'Cluster': 'Cluster'})
    s_ds2 = s_ds2.rename(columns={'f1_freq': 'f1', 'f1_amp_sqrt': 'A1', 'f1_fase': 'P1', 'f2_freq': 'f2', 'f2_amp_sqrt': 'A2', 'arm_first': 'Harm1', 'arm_sec': 'Harm2', 'comb_suma': 'AddComb', 'comb_rest': 'SubComb', 'Cluster': 'Cluster'})
    s_ds_original = s_ds_original[['f1_freq', 'f1_amp_sqrt', 'f1_fase', 'f2_freq', 'f2_amp_sqrt', 'arm_first', 'arm_sec', 'comb_suma', 'comb_rest']].copy()
    s_ds_original = s_ds_original.rename(columns={'f1_freq': 'f1', 'f1_amp_sqrt': 'A1', 'f1_fase': 'P1', 'f2_freq': 'f2', 'f2_amp_sqrt': 'A2', 'arm_first': 'Harm1', 'arm_sec': 'Harm2', 'comb_suma': 'AddComb', 'comb_rest': 'SubComb', 'Cluster': 'Cluster'})
    variables = s_ds_original.columns
    # Loop over the data and labels and plot each variable
    for i, data in enumerate(variables):
        # sns.set(style="whitegrid")
        # Plot the histogram
        # tip = sns.load_dataset("tips")
        # sns.boxplot(y="total_bill", 
        #             hue="smoker",
        #             data=tip, palette="Set2",
        #             dodge=True)

        # Create a boxplot of the data
        
        if boxplots_exp:
            sns.boxplot(y=np.log10(s_ds_original[data]+1), dodge=True, width=0.3, whis=1.5, ax=axs4[i]) # palette="Set3") #y=data, dodge=True, whis=[5,95] (cahnge outliers computation)
            #sns.boxenplot(y=data, data=s_ds_original, dodge=True, width=0.3, ax=axs4[i], scale='exponential') # palette="Set3") #y=data, dodge=True, whis=[5,95] (cahnge outliers computation)
            fig4.suptitle('Boxplots with outliers information, exponential model (log10(data+1))', fontsize=15)
        else:
            sns.boxplot(y=s_ds_original[data], dodge=True, width=0.3, whis=1.5, ax=axs4[i]) # palette="Set3") #y=data, dodge=True, whis=[5,95] (cahnge outliers computation)
            fig4.suptitle('Boxplots with outliers information', fontsize=15)
        
        # Compute the median
        median = np.median(s_ds_original[data])
        # Compute the interquartile range (IQR)
        q1 = np.percentile(s_ds_original[data], 25)
        q3 = np.percentile(s_ds_original[data], 75)
        iqr = q3 - q1
        # Compute the IQD
        iqd = iqr / 2
        outliers_lim_IQR = q3 + 1.5*iqr # Fundamental phase does not have outliers even if the limit is closer than q3+1.5*iqr, so it draws the whiskers closer
        if boxplots_exp:
            # Compute the median for exponential
            median_exp = np.median(np.log10(s_ds_original[data]+1))
            # Compute the interquartile range (IQR)
            q1_exp = np.percentile(np.log10(s_ds_original[data]+1), 25)
            q3_exp = np.percentile(np.log10(s_ds_original[data]+1), 75)
            iqr_exp = q3_exp - q1_exp
            outliers_lim_IQR_exp = q3_exp + 1.5*iqr_exp # Fundamental phase does not have outliers even if the limit is closer than q3+1.5*iqr, so it draws the whiskers closer
            #np.log10(q3 + 1.5*iqr+1)?

        # outliers_lim_IQR = np.percentile(s_ds_original[data], 95)  # For whis=[5,95], when 95th quartile is desired
        # n_outliers = len(s_ds_original.query(data + '>=' + str(outliers_lim_IQR)))

        # Save outliers
        if Save_outliers:
            outliers_dict_temp_IQR = []
            outliers_dict_temp_IQR = s_ds_original.query(data + '>=' + str(outliers_lim_IQR))
            outliers_list_IQR = list(outliers_dict_temp_IQR.index)
            for j in range(len(outliers_list_IQR)):
                outliers_ds_IQR.loc[len(outliers_ds_IQR)] = s_ds_original.iloc[outliers_list_IQR[j]] # Loc: access a row in a dataset, logic saves data of the whole structure in next row
                outliers_ds_IQR.loc[len(outliers_ds_IQR)-1,'outlier_reason_IQR'] = data
        
        # Save outliers
        if Save_outliers and boxplots_exp:
            exp_data_df = pd.DataFrame({'data': np.log10(s_ds_original[data]+1)})
            outliers_dict_temp_IQR = []
            outliers_dict_temp_IQR = exp_data_df.query('data >= @outliers_lim_IQR_exp') #@ to get a variable in a string
            outliers_list_IQR = list(outliers_dict_temp_IQR.index)
            for j in range(len(outliers_list_IQR)):
                outliers_ds_IQR.loc[len(outliers_ds_IQR)] = s_ds_original.iloc[outliers_list_IQR[j]] # Loc: access a row in a dataset, logic saves data of the whole structure in next row
                outliers_ds_IQR.loc[len(outliers_ds_IQR)-1,'outlier_reason_IQR'] = data

        # Add a title and label to the plot
        axs4[i].set_title(titles[i], fontsize=10)
        axs4[i].set_xlabel('Type')
        axs4[i].set_ylabel('Value')
    # Adjust the spacing between the plots
    plt.subplots_adjust(wspace=0.3, hspace=0.6)

    # See repeated stars that are outliers and saving everything
    if Save_outliers:
        list_outliers_IQR = outliers_ds_IQR['Unnamed: 0'].value_counts()
        with open('./new/Output_new/Exploratory_Stats_Analysis/outliers_info_IQR.pkl', 'wb') as f:
            pickle.dump(outliers_ds_IQR, f)
        with open('./new/Output_new/Exploratory_Stats_Analysis/outliers_list_IQR.pkl', 'wb') as f:
            pickle.dump(list_outliers_IQR, f)
        list_outliers_IQR.to_csv('./new/Output_new/Exploratory_Stats_Analysis/Outliers_list_IQR.csv', index=True)
        outliers_ds_IQR.to_csv('./new/Output_new/Exploratory_Stats_Analysis/Outliers_info_IQR.csv', index=True)

    # See repeated stars that are outliers and saving everything
    if Save_outliers and boxplots_exp:
        list_outliers_IQR = outliers_ds_IQR['Unnamed: 0'].value_counts()
        with open('./new/Output_new/Exploratory_Stats_Analysis/outliers_info_IQR_exponential.pkl', 'wb') as f:
            pickle.dump(outliers_ds_IQR, f)
        with open('./new/Output_new/Exploratory_Stats_Analysis/outliers_list_IQR_exponential.pkl', 'wb') as f:
            pickle.dump(list_outliers_IQR, f)
        list_outliers_IQR.to_csv('./new/Output_new/Exploratory_Stats_Analysis/Outliers_list_IQR_exponential.csv', index=True)
        outliers_ds_IQR.to_csv('./new/Output_new/Exploratory_Stats_Analysis/Outliers_info_IQR_exponential.csv', index=True)

    ##### BOXPLOTS HADS LADS
    fig5, axs5 = plt.subplots(3, 3, figsize=(12, 10))
    fig5.suptitle('Boxplots with outliers information HADS and LADS', fontsize=15)
    axs5 = axs5.ravel()
    colors = ['#D55E00', '#0072B2'] # 'colorblind' palette from sns
    sns.set_palette(colors)
    variables2 = s_ds.columns
    for i, data in enumerate(variables2):    
        # Create a boxplot of the data
        sns.boxplot(y=data, x ='type', data=s_ds2, dodge=True, ax=axs5[i]) # palette="Set3") #y=data, dodge=True
        # Add a title and label to the plot
        axs5[i].set_title(titles[i], fontsize=10)
        axs5[i].set_xlabel('Type')
        axs5[i].set_ylabel('Value')
            
    # Adjust the spacing between the plots
    plt.subplots_adjust(wspace=0.3, hspace=0.6)












# Show the plots
if Show_figures:
    plt.show()


if Save_figures:
    # save the figure in png format
    if histograms:
        fig0.savefig('./new/Figures_new/Exploratory_Stats_Analysis/HistogramF1Amp.png', overwrite=overwrite_val)
        fig00.savefig('./new/Figures_new/Exploratory_Stats_Analysis/HistogramComSum.png', overwrite=overwrite_val)
        fig01.savefig('./new/Figures_new/Exploratory_Stats_Analysis/HistogramComRes.png', overwrite=overwrite_val)
        fig.savefig('./new/Figures_new/Exploratory_Stats_Analysis/HistogramF1AmpOutliers.png', overwrite=overwrite_val)
        fig1.savefig('./new/Figures_new/Exploratory_Stats_Analysis/HistogramALL.png', overwrite=overwrite_val)
        fig2.savefig('./new/Figures_new/Exploratory_Stats_Analysis/HistogramHADSLADS.png', overwrite=overwrite_val)
        fig3.savefig('./new/Figures_new/Exploratory_Stats_Analysis/HistogramStatsOutliers.png', overwrite=overwrite_val)
        # save the figure in pdf format
        fig0.savefig('./new/Figures_new/Exploratory_Stats_Analysis/HistogramF1Amp.pdf', overwrite=overwrite_val)
        fig00.savefig('./new/Figures_new/Exploratory_Stats_Analysis/HistogramComSum.pdf', overwrite=overwrite_val)
        fig01.savefig('./new/Figures_new/Exploratory_Stats_Analysis/HistogramComRes.pdf', overwrite=overwrite_val)
        fig.savefig('./new/Figures_new/Exploratory_Stats_Analysis/HistogramF1AmpOutliers.pdf', overwrite=overwrite_val)
        fig1.savefig('./new/Figures_new/Exploratory_Stats_Analysis/HistogramALL.pdf', overwrite=overwrite_val)
        fig2.savefig('./new/Figures_new/Exploratory_Stats_Analysis/HistogramHADSLADS.pdf', overwrite=overwrite_val)
        fig3.savefig('./new/Figures_new/Exploratory_Stats_Analysis/HistogramStatsOutliers.pdf', overwrite=overwrite_val)
    if boxplots_exp and boxplots:
        fig4.savefig('./new/Figures_new/Exploratory_Stats_Analysis/BoxplotsALL_exponential2.png', overwrite=overwrite_val)
        # save the figure in pdf format
        fig4.savefig('./new/Figures_new/Exploratory_Stats_Analysis/BoxplotsALL_exponential2.pdf', overwrite=overwrite_val)

    if boxplots and boxplots_exp==0:
        fig4.savefig('./new/Figures_new/Exploratory_Stats_Analysis/BoxplotsALL.png', overwrite=overwrite_val)
        fig5.savefig('./new/Figures_new/Exploratory_Stats_Analysis/BoxplotsHADSLADS.png', overwrite=overwrite_val)
        # save the figure in pdf format
        fig4.savefig('./new/Figures_new/Exploratory_Stats_Analysis/BoxplotsALL.pdf', overwrite=overwrite_val)
        fig5.savefig('./new/Figures_new/Exploratory_Stats_Analysis/BoxplotsHADSLADS.pdf', overwrite=overwrite_val)


        # strFile = ['./figure' + str(val[i+1]) + '.pdf']
        # if os.path.isfile(strFile) and overwrite_val:
        #     os.remove(strFile)   # Opt.: os.system("rm "+strFile)
        #     figure = ['fig' + str(val[i])]
        #     figure.savefig(strFile)
