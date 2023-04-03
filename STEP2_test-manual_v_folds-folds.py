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
import numpy as np
import seaborn as sns; sns.set_theme()

sns.set(font_scale=2)

def plot_matrix(cm, classes, title, savepath):
  plt.figure(figsize=(10,10))
  axp = sns.heatmap(cm, cmap="Blues", annot=True, xticklabels=classes, yticklabels=classes, cbar=False)
  axp.set(title=title, xlabel="Median Interfold < Threshold", ylabel="Ensemble < Threshold")
  axp.figure.savefig(savepath)

def dice(array1, array2):
	# Convert to boolean
	array1 = array1.astype(bool).copy()
	array2 = array2.astype(bool).copy()

	intersection = np.logical_and(array1, array2)

	return 2. * intersection.sum() / (array1.sum() + array2.sum())

########################'[variables to update]##########################################

# same as STEP1
project_title = ''
root_folder = ''
output_folder = root_folder + project_title + '/'
intermediate_folder = output_folder + '/intermediate_sim_metrics/'

# thresholds to compare summarized interfold dices against - recommend "priors" from human-human interobserver studies
io1 = 0.93 # example
io2 = 0.90

# can input other thresholds for sensitivity test
bad_ensemble_threshold2_1 = 0.93
bad_ensemble_threshold2_2 = 0.95
bad_ensemble_threshold2_3 = 0.98

table1 = pd.read_csv(intermediate_folder + 'LABEL_1' + 'confidence_dice_similarity_metrics.csv')
table2 = pd.read_csv(intermediate_folder + 'LABEL_2' + 'confidence_dice_similarity_metrics.csv')

interfold_v_test1 = pd.read_csv(intermediate_folder + 'comp1.csv')
interfold_v_test2 = pd.read_csv(intermediate_folder + 'comp2.csv')

########################'[code execution]##########################################

table1 = pd.read_csv(intermediate_folder + 'LABEL_1' + 'confidence_dice_similarity_metrics.csv')
table2 = pd.read_csv(intermediate_folder + 'LABEL_2' + 'confidence_dice_similarity_metrics.csv')

interfold_v_test1 = pd.read_csv(intermediate_folder + 'comp1.csv')
interfold_v_test2 = pd.read_csv(intermediate_folder + 'comp2.csv')

sns.set_style("whitegrid")
plt.figure()

p1 = sns.FacetGrid(interfold_v_test1, col="metric")
p1.map(sns.scatterplot, "Dice", "Dice_comp")
p1.set_axis_labels("Test Ensemble Dice", "Interfold Metric")
p1.set(xlim=(0, 1),ylim=(0, 1))
ax1, ax2, ax3, ax4, ax5, ax6 = p1.axes[0]

ax1.axhline(io1, ls='--')
ax2.axhline(io1, ls='--')
ax3.axhline(io1, ls='--')
ax4.axhline(io1, ls='--')

ax1.axvline(io1, ls='--', color = 'r')
ax2.axvline(io1, ls='--', color = 'r')
ax3.axvline(io1, ls='--', color = 'r')
ax4.axvline(io1, ls='--', color = 'r')
ax5.axvline(io1, ls='--', color = 'r')
ax6.axvline(io1, ls='--', color = 'r')

plt.savefig(output_folder + 'Label1.png')

sns.set_style("whitegrid")
plt.figure()

p2 = sns.FacetGrid(interfold_v_test2, col="metric")
p2.map(sns.scatterplot, "Dice", "Dice_comp")
p2.set_axis_labels("Test Ensemble Dice", "Interfold Metric")
p2.set(xlim=(0, 1),ylim=(0, 1))
ax1, ax2, ax3, ax4, ax5, ax6 = p2.axes[0]

ax1.axhline(io2, ls='--')
ax2.axhline(io2, ls='--')
ax3.axhline(io2, ls='--')
ax4.axhline(io2, ls='--')

ax1.axvline(io2, ls='--', color = 'r')
ax2.axvline(io2, ls='--', color = 'r')
ax3.axvline(io2, ls='--', color = 'r')
ax4.axvline(io2, ls='--', color = 'r')
ax5.axvline(io2, ls='--', color = 'r')
ax6.axvline(io2, ls='--', color = 'r')

plt.savefig(output_folder + 'Label2.png')


mets = ['mean','median','max','min']

c_interfold_v_test1 = copy.deepcopy(interfold_v_test1)

table1 = pd.DataFrame({'Method': pd.Series([], dtype ='str'),
								 'Mean All Images': pd.Series([], dtype ='float'),
								 'SD All Images': pd.Series([], dtype ='float'),
								 'N Flag Images': pd.Series([], dtype ='float'),
								 'Mean Flag Images': pd.Series([], dtype ='float'),
								 'SD Flag Images': pd.Series([], dtype ='float'), 
								 'Mean WO Flag Images': pd.Series([], dtype ='float'),
								 'SD WO Flag Images': pd.Series([], dtype ='float')})
table2 = pd.DataFrame({'Method': pd.Series([], dtype ='str'),
								 'Mean All Images': pd.Series([], dtype ='float'),
								 'SD All Images': pd.Series([], dtype ='float'),
								 'N Flag Images': pd.Series([], dtype ='float'),
								 'Mean Flag Images': pd.Series([], dtype ='float'),
								 'SD Flag Images': pd.Series([], dtype ='float'), 
								 'Mean WO Flag Images': pd.Series([], dtype ='float'),
								 'SD WO Flag Images': pd.Series([], dtype ='float')})

print('label1')
for met in mets:
	print(met)
	interfold_v_test1 = copy.deepcopy(c_interfold_v_test1)
	interfold_v_test1 = interfold_v_test1[interfold_v_test1['metric'] == met]
	print(io1)
	mean_all = np.round(interfold_v_test1['Dice'].mean(),3)
	sd_all = np.round(interfold_v_test1['Dice'].std(),3)
	mean_flagged = np.round(interfold_v_test1.loc[interfold_v_test1['Dice_comp']<io1]['Dice'].mean(),3)
	sd_flagged = np.round(interfold_v_test1[interfold_v_test1['Dice_comp']<io1]['Dice'].std(),3)
	mean_wo_flagged = np.round(interfold_v_test1.loc[interfold_v_test1['Dice_comp']>io1]['Dice'].mean(),3)
	sd_wo_flagged = np.round(interfold_v_test1[interfold_v_test1['Dice_comp']>io1]['Dice'].std(),3)
	num_flag = interfold_v_test1[interfold_v_test1['Dice_comp']<io1].shape[0]
	row = {'Method': met, 'Mean All Images':mean_all, 'SD All Images' : sd_all,'N Flag Images': num_flag, 'Mean Flag Images': mean_flagged, 'SD Flag Images': sd_flagged, 'Mean WO Flag Images': mean_wo_flagged, 'SD WO Flag Images': sd_wo_flagged}
	table1 = table1.append(row, ignore_index = True)

c_interfold_v_test2 = copy.deepcopy(interfold_v_test2)

print('label2')
for met in mets:
	print(met)
	interfold_v_test2 = copy.deepcopy(c_interfold_v_test2)
	interfold_v_test2 = interfold_v_test2[interfold_v_test2['metric'] == met]
	print(io2)
	mean_all = np.round(interfold_v_test2['Dice'].mean(),3)
	sd_all = np.round(interfold_v_test2['Dice'].std(),3)
	mean_flagged = np.round(interfold_v_test2.loc[interfold_v_test2['Dice_comp']<io2]['Dice'].mean(),3)
	sd_flagged = np.round(interfold_v_test2[interfold_v_test2['Dice_comp']<io2]['Dice'].std(),3)
	mean_wo_flagged = np.round(interfold_v_test2.loc[interfold_v_test2['Dice_comp']>io2]['Dice'].mean(),3)
	sd_wo_flagged = np.round(interfold_v_test2[interfold_v_test2['Dice_comp']>io2]['Dice'].std(),3)
	num_flag = interfold_v_test2[interfold_v_test2['Dice_comp']<io2].shape[0]
	row = {'Method': met, 'Mean All Images':mean_all, 'SD All Images' : sd_all,'N Flag Images': num_flag, 'Mean Flag Images': mean_flagged, 'SD Flag Images': sd_flagged, 'Mean WO Flag Images': mean_wo_flagged, 'SD WO Flag Images': sd_wo_flagged}
	table2 = table2.append(row, ignore_index = True)

table1.to_csv(output_folder + 'table1.csv')
table2.to_csv(output_folder + 'table2.csv')

# confusion matrix

conf_mat1 = pd.read_csv(intermediate_folder + 'comp1.csv')
conf_mat2 = pd.read_csv(intermediate_folder + 'comp2.csv')

conf_mat1 = conf_mat1.loc[conf_mat1['metric'] == 'median']
conf_mat2 = conf_mat2.loc[conf_mat2['metric'] == 'median']

# true positives - where ensemble < threshold and < interfold
tp1 = conf_mat1.loc[(conf_mat1['Dice'] < bad_ensemble_threshold1) & (conf_mat1['Dice_comp'] < io1)].shape[0]
tp2 = conf_mat2.loc[(conf_mat2['Dice'] < bad_ensemble_threshold2) & (conf_mat2['Dice_comp'] < io2)].shape[0]

# false positives - where ensemble > threshold and < interfold
fp1 = conf_mat1.loc[(conf_mat1['Dice'] < bad_ensemble_threshold1) & (conf_mat1['Dice_comp'] > io1)].shape[0]
fp2 = conf_mat2.loc[(conf_mat2['Dice'] < bad_ensemble_threshold2) & (conf_mat2['Dice_comp'] > io2)].shape[0]

# false negative - where ensemble < threshold and > interfold
fn1 = conf_mat1.loc[(conf_mat1['Dice'] > bad_ensemble_threshold1) & (conf_mat1['Dice_comp'] < io1)].shape[0]
fn2 = conf_mat2.loc[(conf_mat2['Dice'] > bad_ensemble_threshold2) & (conf_mat2['Dice_comp'] < io2)].shape[0]

# true negative - where ensemble > threshold and > interfold
tn1 = conf_mat1.loc[(conf_mat1['Dice'] > bad_ensemble_threshold1) & (conf_mat1['Dice_comp'] > io1)].shape[0]
tn2 = conf_mat2.loc[(conf_mat2['Dice'] > bad_ensemble_threshold2) & (conf_mat2['Dice_comp'] > io2)].shape[0]

# Plot confusion matricies

classes = ['Positive', 'Negative']

cm1 = np.array([[tp1, fp1], [fn1, tn1]])
title1 = project_title + ' label_1 \n true label + if ensemble > ' + str(bad_ensemble_threshold1) + '\n predicted label + if median of interfold > ' + str(io1)

cm2 = np.array([[tp2, fp2], [fn2, tn2]])
title2 = project_title + ' label_2 \n true label + if ensemble > ' + str(bad_ensemble_threshold2) + '\n predicted label + if median of interfold > ' + str(io2)

plot_matrix(cm1, classes, title1, output_folder + 'cm1.png')
plot_matrix(cm2, classes, title2, output_folder + 'cm2.png')

# Test different thresholds

# # true positives - where ensemble < threshold and < interfold
tp2_1 = conf_mat1.loc[(conf_mat1['Dice'] < io1) & (conf_mat1['Dice_comp'] < bad_ensemble_threshold2_1)].shape[0]
tp2_2 = conf_mat1.loc[(conf_mat1['Dice'] < io1) & (conf_mat1['Dice_comp'] < bad_ensemble_threshold2_2)].shape[0]
tp2_3 = conf_mat1.loc[(conf_mat1['Dice'] < io1) & (conf_mat1['Dice_comp'] < bad_ensemble_threshold2_3)].shape[0]

# # false positives - where ensemble > threshold and < interfoldfp1 = conf_mat1.loc[(conf_mat1['Dice'] < bad_ensemble_threshold1) & (conf_mat1['Dice_comp'] > io1)].shape[0]
fp2_1 = conf_mat1.loc[(conf_mat1['Dice'] < io1) & (conf_mat1['Dice_comp'] > bad_ensemble_threshold2_1)].shape[0]
fp2_2 = conf_mat1.loc[(conf_mat1['Dice'] < io1) & (conf_mat1['Dice_comp'] > bad_ensemble_threshold2_2)].shape[0]
fp2_3 = conf_mat1.loc[(conf_mat1['Dice'] < io1) & (conf_mat1['Dice_comp'] > bad_ensemble_threshold2_3)].shape[0]

# # false negative - where ensemble < threshold and > interfold
fn2_1 = conf_mat1.loc[(conf_mat1['Dice'] > io1) & (conf_mat1['Dice_comp'] < bad_ensemble_threshold2_1)].shape[0]
fn2_2 = conf_mat1.loc[(conf_mat1['Dice'] > io1) & (conf_mat1['Dice_comp'] < bad_ensemble_threshold2_2)].shape[0]
fn2_3 = conf_mat1.loc[(conf_mat1['Dice'] > io1) & (conf_mat1['Dice_comp'] < bad_ensemble_threshold2_3)].shape[0]

# # true negative - where ensemble > threshold and > interfold
tn2_1 = conf_mat1.loc[(conf_mat1['Dice'] > io1) & (conf_mat1['Dice_comp'] > bad_ensemble_threshold2_1)].shape[0]
tn2_2 = conf_mat1.loc[(conf_mat1['Dice'] > io1) & (conf_mat1['Dice_comp'] > bad_ensemble_threshold2_2)].shape[0]
tn2_3 = conf_mat1.loc[(conf_mat1['Dice'] > io1) & (conf_mat1['Dice_comp'] > bad_ensemble_threshold2_3)].shape[0]

cm2_1 = np.array([[tp2_1, fp2_1], [fn2_1, tn2_1]])
cm2_2 = np.array([[tp2_2, fp2_2], [fn2_2, tn2_2]])
cm2_3 = np.array([[tp2_3, fp2_3], [fn2_3, tn2_3]])
# title2 = ''

plot_matrix(cm2_1, classes, str(bad_ensemble_threshold2_1), output_folder + 'cm2_1.png')
plot_matrix(cm2_2, classes, str(bad_ensemble_threshold2_2), output_folder + 'cm2_2.png')
plot_matrix(cm2_3, classes, str(bad_ensemble_threshold2_3), output_folder + 'cm2_3.png')


