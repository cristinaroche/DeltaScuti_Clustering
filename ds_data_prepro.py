import numpy as np
import pandas as pd
import glob
import os 
import csv
import pickle

# Load data
data_folder = './new/para_cris_todas_new/para_cris'
all_stars_files = glob.glob(data_folder + '/' + '*.dat') # glob to match all the .dat files in the directory
stars_dict = {} # empty outer dictionary to store the original filenames (rows)
stars_dict_reader = {} # Create a key for each stars csv file appart, since csv can't be exported. Outer dictionary

# loop through every file found in data_folder with extension .dat
# Creates an inner dictionary for each stars_dict[var_name] dictionary and saves the data in keys
for i, filename in enumerate(all_stars_files):
    armonics_first = 0 # reset values for every star
    armonics_sec = 0
    combinations_suma  = 0
    combinations_resta = 0 
    second_mode = False # it only changes to true if there is second mode in that star

    # Define the variable name string based on the filename, save it as the key of the outer dictionary
    var_name = 'star_' + str(i)
    stars_dict[var_name] = {} # Create a key for each stars with easier names. Inner dictionary (columns keys)
    stars_dict_reader[var_name] = {} 

    # Store the original filename in the dictionary stars_dict in the key 'true_name'
    stars_dict[var_name]['true_name'] = filename
    part_to_remove_f = '_significativas.dat'
    part_to_remove_i = data_folder + '\\'
    index_f = stars_dict[var_name]['true_name'].find(part_to_remove_f)
    index_i = stars_dict[var_name]['true_name'].find(part_to_remove_i)
    stars_dict[var_name]['true_name'] = stars_dict[var_name]['true_name'][:index_f] + stars_dict[var_name]['true_name'][index_f+len(part_to_remove_f):]
    stars_dict[var_name]['true_name'] = stars_dict[var_name]['true_name'][index_i+len(part_to_remove_i):]

    # Fill up key 'type' with the type of star: #0 not type found, #1 HADS, #2 LADS
    hads_type = 1
    lads_type = 2

    stars_dict[var_name]['type'] = 0 
    with open('new/HADS.txt', 'r') as f:
        for line in f:
            if stars_dict[var_name]['true_name'] in line:
                stars_dict[var_name]['type'] = hads_type
    with open('new/LADS.txt', 'r') as f:
        for line in f:
            if stars_dict[var_name]['true_name'] in line:
                stars_dict[var_name]['type'] = lads_type


    # Fill up numerical keys with the contents of the variables of each file, using reader
    # It loops every row of the text file and search for keys, if key found, it enter in the row and saves the desired data of the columns, then goes to the next row
    with open(filename, 'r') as f:
        stars_dict_reader[var_name]['reader'] = csv.reader(f,delimiter=' ')
        for row in stars_dict_reader[var_name]['reader']: # goes through every row of one star
            # Extract and fetch data row[0] checks first column on that row

            # Fundamental frequency data
            if row[0] == 'f1':
                Fundamental_freq = float(row[1])
                Fundamental_amp  = float(row[2])
                Fundamental_fase  = float(row[4])
                #print('Fundamental frequency:', Fundamental_freq)
                #print('Fundamental amplitude:', Fundamental_amp)
                stars_dict[var_name]['f1_freq'] = Fundamental_freq
                stars_dict[var_name]['f1_amp'] = Fundamental_amp
                stars_dict[var_name]['f1_amp_sqrt'] = np.sqrt(Fundamental_amp)
                stars_dict[var_name]['f1_fase'] = Fundamental_fase

            # First overtone data, if no first overtone, its data is 'False'
            if row[0] == 'f2':
                o1_freq = float(row[1])
                o1_amp  = float(row[2])
                #print('First overtone frequency:', o1_freq)
                #print('First overtone amplitude:', o1_amp)
                stars_dict[var_name]['f2_freq'] = o1_freq
                stars_dict[var_name]['f2_amp'] = o1_amp   
                stars_dict[var_name]['f2_amp_sqrt'] = np.sqrt(o1_amp)
                second_mode = True

            # Find armonics: first and/or second digit have to be numbers, leftover string has to be only 'f1' or 'f2' to avoid counting combinations
            if row[0][0].isdigit() and (row[0][1:] == 'f1' or row[0][2:] == 'f1'):
                armonics_first += 1
            if row[0][0].isdigit() and (row[0][1:] == 'f2' or row[0][2:] == 'f2'):
                armonics_sec += 1
            
            # Find plus/minus combinations, just checks if - appears in the first column in that row
            if "+" in row[0]:
                combinations_suma += 1
            if "-" in row[0]:
                combinations_resta += 1

            # Finds nan and prints in the terminal the reference name of the star
            if 'NaN' in row:
                print(var_name)
                print('NaN value found for star with ref: ' + stars_dict[var_name]['true_name'])
            


        # After going through all of the rows, save number of armonics and combinations
        stars_dict[var_name]['arm_first'] = armonics_first
        stars_dict[var_name]['arm_sec'] = armonics_sec
        stars_dict[var_name]['comb_suma'] = combinations_suma      
        stars_dict[var_name]['comb_rest'] = combinations_resta

        # Save as 0 in case after going through the rows there was not 
        if second_mode == False:
            stars_dict[var_name]['f2_freq'] = 0
            stars_dict[var_name]['f2_amp'] = 0
            stars_dict[var_name]['f2_amp_sqrt'] = 0
            stars_dict[var_name]['multiperiodic'] = 0
        else:
            stars_dict[var_name]['multiperiodic'] = 1  

        # Create cathegorical non linear variables
        if combinations_suma == 0:
            stars_dict[var_name]['comb_suma_cat'] = 0
        elif combinations_suma <= 2:
            stars_dict[var_name]['comb_suma_cat'] = 1
        else:
            stars_dict[var_name]['comb_suma_cat'] = 2

        if combinations_resta == 0:
            stars_dict[var_name]['comb_resta_cat'] = 0
        elif combinations_resta <= 2:
            stars_dict[var_name]['comb_resta_cat'] = 1
        elif combinations_resta <= 5:
            stars_dict[var_name]['comb_resta_cat'] = 2
        else:
            stars_dict[var_name]['comb_resta_cat'] = 3

        if armonics_first == 0:
            stars_dict[var_name]['armonics_first_cat'] = 0
        elif armonics_first <= 2:
            stars_dict[var_name]['armonics_first_cat'] = 1
        elif armonics_first <= 5:
            stars_dict[var_name]['armonics_first_cat'] = 2
        else:
            stars_dict[var_name]['armonics_first_cat'] = 2

            

# Check that all stars were found in the list of lads and hads
contador_lads = 0
contador_hads = 0
contador_notype = 0
n_lads = []
for case in stars_dict.keys():
    if stars_dict[case]['type'] == hads_type:
        contador_hads = contador_hads +1
    elif stars_dict[case]['type'] == lads_type:
        contador_lads = contador_lads +1
    else:
        contador_notype = contador_notype +1
        print(case)
        print('No type found for star with ref: ' + stars_dict[case]['true_name'])

# Save dictionary externally 
with open('stars_dict.pickle', 'wb') as dict_stars_data_processed:
    pickle.dump(stars_dict, dict_stars_data_processed)

# Convert dictionary into a pandas dataset: 
# # Columns: keys of outer dictionary (var_name)
# # Rows: keys of inner dictionary (f1, a1...)
stars_dataset =  pd.DataFrame.from_dict(stars_dict,orient='index')
print(stars_dataset.head())
# Save the dataframe to a CSV file
stars_dataset.to_csv('stars_dataset_new_cathegorical.csv', index=True)


# Tricks to check things in dictionaries:
# For debugging a key
#num_list = []
#for case in stars_dict.keys():
#    num_list.append(stars_dict[case]['f2_amp'])
# if stars_dict[var_name]['true_name'] == './para_cris\\197759259_significativas.dat':
#     print('cris mira mira!')