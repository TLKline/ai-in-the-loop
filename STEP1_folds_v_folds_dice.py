# run interactively or update variables and run at command line
# calculate and summarize interfold dices

#packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import time
import seaborn as sns
import nibabel as nib
import math
import os
import copy
from functools import partial, reduce
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def dice(array1, array2):
	# Convert to boolean
	array1 = array1.astype(bool).copy()
	array2 = array2.astype(bool).copy()

	intersection = np.logical_and(array1, array2)

	return 2. * intersection.sum() / (array1.sum() + array2.sum())

########################'[variables to update]##########################################

dir_fold0_predictions = ''
dir_fold1_predictions = ''
dir_fold2_predictions = ''
dir_fold3_predictions = ''
dir_fold4_predictions = ''

#from STEP0
file_test_v_manual_dice_label1 = ''
file_test_v_manual_dice_label2 = ''

# to set up folder system
project_title = ''
root_folder = ''
output_folder = root_folder + project_title + '/'
intermediate_folder = output_folder + '/intermediate_sim_metrics/'

if not os.path.exists(output_folder):
	os.mkdir(output_folder)
	os.mkdir(intermediate_folder)

# thresholds to compare summarized interfold dices against - recommend "priors" from human-human interobserver studies
io1 = 0.93 # example
io2 = 0.90

# can input other thresholds for sensitivity test
bad_ensemble_threshold2_1 = 0.93
bad_ensemble_threshold2_2 = 0.95
bad_ensemble_threshold2_3 = 0.98

# suffix to remove from 'ID' column in fold simlarity metrics so can merge on 'ID' between interfold csv and test csv
remove_suffix = '.nii.gz'

###############################3'[code execution]########################################3

# folders
interfold_test_prediction_
folders = [ dir_fold0_predictions, dir_fold1_predictions, dir_fold2_predictions, dir_fold3_predictions, dir_fold4_predictions]
label_1_test_ensemble_sim_metrics = pd.read_csv(file_test_v_manual_dice_label1)
label_2_test_ensemble_sim_metrics = pd.read_csv(file_test_v_manual_dice_label2)

bad_ensemble_threshold1 = io1
bad_ensemble_threshold2 = io2

#1) CALCULATE INTERFOLD DICE AND WRITE OUT SUMMARY TABLE


#table for each label
table1 = pd.DataFrame({'ID': pd.Series([], dtype ='str'),
								 'Dice_comp': pd.Series([], dtype ='float'),
								 'pred_folder0': pd.Series([], dtype ='str'),
								 'pred_folder1': pd.Series([], dtype ='str')})
table2 = pd.DataFrame({'ID': pd.Series([], dtype ='str'),
								 'Dice_comp': pd.Series([], dtype ='float'),
								 'pred_folder0': pd.Series([], dtype ='str'),
								 'pred_folder1': pd.Series([], dtype ='str')})

comparisons = combinations(interfold_test_prediction_folders, 2)
n_comp = 0

for comp in comparisons:

	print('on comp # ' + str(n_comp))

	n_comp+=1

	pred0 = comp[0]
	pred1 = comp[1]

	print(pred0)
	print(pred1)

 	# load in arrays
	subdir, dirs, all_files = os.walk(pred0).__next__()
	all_files = [f for f in all_files if remove_suffix in f]
	strremove = -len(remove_suffix)

	for file in all_files:
		
		prediction0_seg = nib.load(pred0 + '/' + file)
		prediction0_affine = prediction0_seg.affine
		prediction0_seg_data = prediction0_seg.get_fdata()

		prediction1_seg = nib.load(pred1 + '/' + file)
		prediction1_affine = prediction1_seg.affine
		prediction1_seg_data = prediction1_seg.get_fdata()


		labels = np.unique(prediction1_seg_data)
		labels = [label for label in labels if label != 0]

		for label in labels:

			label_prediction_seg_data = copy.deepcopy(prediction0_seg_data)
			label_reference_seg_data = copy.deepcopy(prediction1_seg_data)

			label_prediction_seg_data[label_prediction_seg_data != label] = 0
			label_prediction_seg_data[label_prediction_seg_data == label] = 1

			label_reference_seg_data[label_reference_seg_data!= label] = 0
			label_reference_seg_data[label_reference_seg_data== label] = 1

			dice_score = dice(label_prediction_seg_data, label_reference_seg_data)

			# update table
			row = {'ID':file[:strremove], 'Dice_comp' :dice_score,'pred_folder0':pred0, 'pred_folder1':pred1}
			
			if label==1:
				table1 = table1.append(row, ignore_index = True)
			if label ==2:
				table2 = table2.append(row, ignore_index = True)

table1.to_csv(intermediate_folder + 'LABEL_1' + 'confidence_dice_similarity_metrics.csv')
table2.to_csv(intermediate_folder + 'LABEL_2' + 'confidence_dice_similarity_metrics.csv')

#mean
mean1 = table1.groupby('ID')['Dice_comp'].mean().to_frame()
mean1['metric'] = 'mean'
mean2 = table2.groupby('ID')['Dice_comp'].mean().to_frame()
mean2['metric'] = 'mean'
#median
median1 = table1.groupby('ID')['Dice_comp'].median().to_frame()
median1['metric'] = 'median'
median2 = table2.groupby('ID')['Dice_comp'].median().to_frame()
median2['metric'] = 'median'
#min
min1 = table1.groupby('ID')['Dice_comp'].min().to_frame()
min1['metric'] = 'min'
min2 = table2.groupby('ID')['Dice_comp'].min().to_frame()
min2['metric'] = 'min'
#max
max1 = table1.groupby('ID')['Dice_comp'].max().to_frame()
max1['metric'] = 'max'
max2 = table2.groupby('ID')['Dice_comp'].max().to_frame()
max2['metric'] = 'max'
#std
std1 = table1.groupby('ID')['Dice_comp'].std().to_frame()
std1['metric'] = 'std'
std2 = table2.groupby('ID')['Dice_comp'].std().to_frame()
std2['metric'] = 'std'
#range
range1 = table1.groupby('ID')['Dice_comp'].max() - table1.groupby('ID')['Dice_comp'].min()
range1 = range1.to_frame()
range1['metric'] = 'range'
range2 = table2.groupby('ID')['Dice_comp'].max() - table2.groupby('ID')['Dice_comp'].min()
range2 = range2.to_frame()
range2['metric'] = 'range'

df1 = pd.concat([mean1, median1, max1, min1, std1, range1], axis=0)
df2 = pd.concat([mean2, median2, max2, min2, std2, range2], axis=0)

label_1_test_ensemble_sim_metrics = label_1_test_ensemble_sim_metrics[['ID','Dice']]
label_2_test_ensemble_sim_metrics = label_2_test_ensemble_sim_metrics[['ID','Dice']]

interfold_v_test1 = label_1_test_ensemble_sim_metrics.merge(df1, on = 'ID', how = 'right')
interfold_v_test2 = label_2_test_ensemble_sim_metrics.merge(df2, on = 'ID', how = 'right')

interfold_v_test1.to_csv(intermediate_folder + 'comp1.csv')
interfold_v_test2.to_csv(intermediate_folder + 'comp2.csv')
