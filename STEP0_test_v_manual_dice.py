# run on command line
# ensure 218-229 code is appropriate to correctly compute volumes based on original affines
# image_type refers to ^ to calculate volume from affine

# example call
# python similarity_metrics.py --prediction_data_folder dir_ensemble_predictions --prediction_suffix suffix_prediction_files --reference_standard_data_folder dir_manual_segmentations --reference_standard_suffix suffix_manual_segmentation_files --out_folder dir_dice_ensemble-manual --out_file file_name.csv --image_type MR --add_bin 0 


# packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import time
import nibabel as nib
import math
import os
import copy
import argparse
import cv2
from tqdm import *
from scipy.ndimage import binary_erosion, binary_dilation, grey_erosion, grey_dilation, distance_transform_edt, morphology

def dice(array1, array2):
	"""
	Computes the Dice coefficient, a measure of set similarity.

	Parameters
	----------
	array1 : array-like, dtype=bool
		Any array of arbitrary size.  If not boolean, will be converted.
	array2 : array-like, dtype=bool, same shape as array1
		Any other array of identical size.  If not boolean, will be converted.

	Returns
	-------
	dice : float
		Dice's coefficient as a float on range [0,1].
		Maximum similarity = 1
		Least similarity = 0

	"""

	# Convert to boolean
	array1 = array1.astype(bool).copy()
	array2 = array2.astype(bool).copy()

	intersection = np.logical_and(array1, array2)

	return 2. * intersection.sum() / (array1.sum() + array2.sum())

def jaccard(array1, array2):
	"""
	The Jaccard metric for comparing set similarity.

	Parameters
	----------
	array1 : array-like, dtype=bool
		Any array of arbitrary size.  If not boolean, will be converted.
	array2 : array-like, dtype=bool, same shape as array1
		Any other array of identical size.  If not boolean, will be converted.

	Returns
	-------
	jaccard : float
		Jaccard metric returned is a float on range [0,1].
		Maximum similarity = 1
		Least similarity = 0

	"""
	intersection = np.logical_and(array1.astype(np.bool),
								  array2.astype(np.bool))

	union = np.logical_or(array1.astype(np.bool), array2.astype(np.bool))

	return intersection.sum() / union.sum().astype(np.float)

def ppv(test, standard):
	"""
	positive predictive value

	true positives / (true positives + false positives)
	"""
	tp = np.sum(np.logical_and(test == 1, standard == 1))
	fp = np.sum(np.logical_and(test == 1, standard == 0))

	if (tp+fp) == 0:
		return 0
	else:
		return(tp / (tp + fp))


def tpr(test, standard):
	"""
	True Positive Fraction (TPF, sensitivity) binary comparison.

	"""
	tp = np.sum(np.logical_and(test == 1, standard == 1))
	fn = np.sum(np.logical_and(test == 0, standard == 1))
	
	return (tp / (tp + fn))

def spc(test, standard):

	"""
	Specificity

	"""
	tn = np.sum(np.logical_and(test == 0, standard == 0))
	fp = np.sum(np.logical_and(test == 1, standard == 0))

	return(tn / (tn + fp))

def dmean(test, standard):

	kernel = np.array( [[[0,0,0], [0,1,0], [0,0,0]], [[0,1,0], [1,1,1], [0,1,0]], [[0,0,0], [0,1,0], [0,0,0]]], dtype='uint8')
	test_erode = binary_erosion(test, structure=kernel)
	test_edge = test - test_erode

	standard_erode = binary_erosion(standard, structure=kernel)
	standard_edge = standard - standard_erode

	standard_dist = distance_transform_edt(1*np.logical_not(standard_edge))

	d = standard_dist * test_edge
	dmean = np.mean(d)
	return dmean

def surfd(test, standard, affine, connectivity=1):
	
	input_1 = np.atleast_1d(test.astype(np.bool))
	input_2 = np.atleast_1d(standard.astype(np.bool))
	

	conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

	S = (input_1.astype('uint8') - (morphology.binary_erosion(input_1, conn).astype('uint8'))).astype('bool')
	Sprime = (input_2.astype('uint8') - (morphology.binary_erosion(input_2, conn))).astype('uint8').astype('bool')

	
	dta = distance_transform_edt(~S,affine)
	dtb = distance_transform_edt(~Sprime,affine)
	
	sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])
	   
	return sds



def create_sim_metrics(prediction_data_folder, prediction_suffix, reference_standard_data_folder, reference_standard_suffix, out_folder, out_filename, image_type, add_bin):

	# load in arrays
	subdir, dirs, all_files = os.walk(prediction_data_folder).__next__()
	strremove = -len(prediction_suffix)
	# create table 
	table1 = pd.DataFrame({'ID': pd.Series([], dtype ='str'),
									 'Dice': pd.Series([], dtype ='float'),
									 'Jacc': pd.Series([], dtype ='float'),
									 'TPR': pd.Series([], dtype ='float'),
									 'SPC': pd.Series([],dtype = 'float'),
									 'PPV': pd.Series([],dtype = 'float'),
									 'MeanSurfDist': pd.Series([],dtype = 'float'),
									 'ResidualMeanSquareDist': pd.Series([],dtype = 'float'),
									 'HausdorfDist': pd.Series([],dtype = 'float'),
									 'voxDiff': pd.Series([], dtype ='float'),
									 'absvoxDiff': pd.Series([], dtype ='float'),
									 'ARVC': pd.Series([], dtype ='float'),
									 'RefVol': pd.Series([], dtype ='float'),
									 'PredVol': pd.Series([], dtype ='float'),
									 'dmean': pd.Series([], dtype ='float'),
									 'prediction_data_folder': pd.Series([], dtype ='str'),
									 'reference_standard_folder': pd.Series([], dtype ='str')})
	table2 = pd.DataFrame({'ID': pd.Series([], dtype ='str'),
								 'Dice': pd.Series([], dtype ='float'),
								 'Jacc': pd.Series([], dtype ='float'),
								 'TPR': pd.Series([], dtype ='float'),
								 'SPC': pd.Series([],dtype = 'float'),
								 'PPV': pd.Series([],dtype = 'float'),
								 'MeanSurfDist': pd.Series([],dtype = 'float'),
								 'ResidualMeanSquareDist': pd.Series([],dtype = 'float'),
								 'HausdorfDist': pd.Series([],dtype = 'float'),
								 'voxDiff': pd.Series([], dtype ='float'),
								 'absvoxDiff': pd.Series([], dtype ='float'),
								 'ARVC': pd.Series([], dtype ='float'),
								 'RefVol': pd.Series([], dtype ='float'),
								 'PredVol': pd.Series([], dtype ='float'),
								 'dmean': pd.Series([], dtype ='float'),
								 'prediction_data_folder': pd.Series([], dtype ='str'),
								 'reference_standard_folder': pd.Series([], dtype ='str')})
	table3 = pd.DataFrame({'ID': pd.Series([], dtype ='str'),
										 'Dice': pd.Series([], dtype ='float'),
										 'Jacc': pd.Series([], dtype ='float'),
										 'TPR': pd.Series([], dtype ='float'),
										 'SPC': pd.Series([],dtype = 'float'),
										 'PPV': pd.Series([],dtype = 'float'),
										 'MeanSurfDist': pd.Series([],dtype = 'float'),
										 'ResidualMeanSquareDist': pd.Series([],dtype = 'float'),
										 'HausdorfDist': pd.Series([],dtype = 'float'),
										 'voxDiff': pd.Series([], dtype ='float'),
										 'absvoxDiff': pd.Series([], dtype ='float'),
										 'ARVC': pd.Series([], dtype ='float'),
										 'RefVol': pd.Series([], dtype ='float'),
										 'PredVol': pd.Series([], dtype ='float'),
										 'dmean': pd.Series([], dtype ='float'),
										 'prediction_data_folder': pd.Series([], dtype ='str'),
										 'reference_standard_folder': pd.Series([], dtype ='str')})
	
	all_files = [f for f in all_files if '.nii.gz' in f]
	for file in tqdm(all_files):
		print(prediction_data_folder + file)
		print(reference_standard_data_folder +file[:strremove] + reference_standard_suffix)
		try:

			prediction_seg = nib.load(prediction_data_folder + file)
			prediction_affine = prediction_seg.affine
			prediction_seg_data = prediction_seg.get_fdata()
			print(prediction_affine)
			reference_seg = nib.load(reference_standard_data_folder +file[:strremove] + reference_standard_suffix)
			reference_affine = reference_seg.affine
			reference_seg_data = reference_seg.get_fdata()
			print(reference_affine)

			if image_type == "MR":
				x = np.abs(reference_affine[0][0])
				y = np.abs(reference_affine[2][1])
				z = np.abs(reference_affine[1][2])
			elif image_type == "CT":
				x = np.abs(reference_affine[0][0])
				y = np.abs(reference_affine[1][1])
				z = np.abs(reference_affine[2][2])
			elif image_type == "CT_KITS":
				x = np.abs(reference_affine[0][2])
				y = np.abs(reference_affine[1][1])
				z = np.abs(reference_affine[2][0])

			labels = np.unique(reference_seg_data)
			labels = [label for label in labels if label != 0]
			print(labels)

			for label in labels:
				print(label)
				label_prediction_seg_data = copy.deepcopy(prediction_seg_data)
				label_reference_seg_data = copy.deepcopy(reference_seg_data)

				label_prediction_seg_data[label_prediction_seg_data != label] = 0
				label_prediction_seg_data[label_prediction_seg_data == label] = 1

				label_reference_seg_data[label_reference_seg_data!= label] = 0
				label_reference_seg_data[label_reference_seg_data== label] = 1

				dice_score = dice(label_prediction_seg_data, label_reference_seg_data)
				jaccard_score = jaccard(label_prediction_seg_data, label_reference_seg_data)
				tpr_score = tpr(label_prediction_seg_data, label_reference_seg_data)
				ppv_score = ppv(label_prediction_seg_data, label_reference_seg_data)
				spc_score = spc(label_prediction_seg_data, label_reference_seg_data)
				dmean_score = dmean(label_prediction_seg_data, label_reference_seg_data)
				surface_distance = surfd(label_prediction_seg_data, label_reference_seg_data, [1, 1, 1],1)
				msd_score = surface_distance.mean()
				rms_score = np.sqrt((surface_distance**2).mean())
				hd_score  = surface_distance.max()

				# volumes
				pred_vox = abs(np.sum(label_prediction_seg_data))
				ref_vox = abs(np.sum(label_reference_seg_data))
				voxDiff = pred_vox - ref_vox
				absvoxDiff = abs(voxDiff)
				ARVC = absvoxDiff / (np.sum(label_reference_seg_data))

				pred_vol_ml = (pred_vox * x * y * z /1000)
				ref_vol_ml = (ref_vox * x * y * z /1000)
				volDiff = pred_vol_ml - ref_vol_ml

				# update table
				row = {'ID':file[:strremove], 'Dice' :dice_score, 'Jacc':jaccard_score, 'TPR': tpr_score, 'PPV':ppv_score, 'ARVC':ARVC, 'dmean':dmean_score, 'SPC':spc_score,'MeanSurfDist':msd_score,'ResidualMeanSquareDist': rms_score,'HausdorfDist':hd_score, 'voxDiff':voxDiff, 'absvoxDiff':absvoxDiff, 'RefVol':ref_vol_ml, 'PredVol': pred_vol_ml, 'volDiff' :volDiff, 'prediction_data_folder':prediction_data_folder, 'reference_standard_folder':reference_standard_data_folder}
				
				if label==1:
					table1 = table1.append(row, ignore_index = True)
				if label ==2:
					table2 = table2.append(row, ignore_index = True)

			if int(add_bin) == 1:
				print('bin')

				label_prediction_seg = nib.load(prediction_data_folder + file)
				label_prediction_affine = label_prediction_seg.affine
				label_prediction_seg_data = label_prediction_seg.get_fdata()

				label_reference_seg = nib.load(reference_standard_data_folder +file[:strremove] + reference_standard_suffix)
				label_reference_affine = label_reference_seg.affine
				label_reference_seg_data = label_reference_seg.get_fdata()

				if image_type == "MR":
						x = np.abs(label_reference_affine[0][0])
						y = np.abs(label_reference_affine[2][1])
						z = np.abs(label_reference_affine[1][2])
				elif image_type == "CT":
						x = np.abs(label_reference_affine[0][0])
						y = np.abs(label_reference_affine[1][1])
						z = np.abs(label_reference_affine[2][2])

				print(x,y,z)
				print(np.unique(label_prediction_seg_data))

				label_prediction_seg_data[label_prediction_seg_data > 0 ] = 1
				label_reference_seg_data[label_reference_seg_data >0] = 1

				print(np.unique(label_prediction_seg_data))

				dice_score = dice(label_prediction_seg_data, label_reference_seg_data)
				jaccard_score = jaccard(label_prediction_seg_data, label_reference_seg_data)
				tpr_score = tpr(label_prediction_seg_data, label_reference_seg_data)
				ppv_score = ppv(label_prediction_seg_data, label_reference_seg_data)
				spc_score = spc(label_prediction_seg_data, label_reference_seg_data)
				dmean_score = dmean(label_prediction_seg_data, label_reference_seg_data)
				surface_distance = surfd(label_prediction_seg_data, label_reference_seg_data, [1, 1, 1],1)
				msd_score = surface_distance.mean()
				rms_score = np.sqrt((surface_distance**2).mean())
				hd_score  = surface_distance.max()

				# volumes
				pred_vox = abs(np.sum(label_prediction_seg_data))
				ref_vox = abs(np.sum(label_reference_seg_data))
				voxDiff = pred_vox - ref_vox
				absvoxDiff = abs(voxDiff)
				ARVC = absvoxDiff / (np.sum(label_reference_seg_data))

				pred_vol_ml = (pred_vox * x * y * z /1000)
				ref_vol_ml = (ref_vox * x * y * z /1000)
				volDiff = pred_vol_ml - ref_vol_ml

				# update table
				row = {'ID':file[:strremove], 'Dice' :dice_score, 'Jacc':jaccard_score, 'TPR': tpr_score, 'PPV':ppv_score, 'ARVC':ARVC, 'dmean':dmean_score, 'SPC':spc_score,'MeanSurfDist':msd_score,'ResidualMeanSquareDist': rms_score,'HausdorfDist':hd_score, 'voxDiff':voxDiff, 'absvoxDiff':absvoxDiff, 'RefVol':ref_vol_ml, 'PredVol': pred_vol_ml, 'volDiff' :volDiff, 'prediction_data_folder':prediction_data_folder, 'reference_standard_folder':reference_standard_data_folder}
				
				table3 = table3.append(row, ignore_index = True)
		except:
			print(file)

		table1.to_csv(out_folder + 'LABEL_1' + '_similarity_metrics_' + out_filename)
		table2.to_csv(out_folder + 'LABEL_2' + '_similarity_metrics_' + out_filename)

		if int(add_bin) == 1:
			table3.to_csv(out_folder + 'LABEL_BIN' + '_similarity_metrics_' + out_filename)

def main(argv):

	create_sim_metrics(argv.prediction_data_folder, argv.prediction_suffix, argv.reference_standard_data_folder, argv.reference_standard_suffix,
		argv.out_folder, argv.out_filename, argv.image_type, argv.add_bin)

if __name__ == "__main__":

	parser = argparse.ArgumentParser( description='This is program to compute similarity metrics per segmentation')
	parser.add_argument ("--prediction_data_folder",  help="folder of predicitions", required=True )
	parser.add_argument ("--prediction_suffix",  help="suffix of predictions ie model_name", required=True )
	parser.add_argument ("--reference_standard_data_folder",  help="folder of reference segmentations", required=True )
	parser.add_argument ("--reference_standard_suffix",  help="suffix of reference segmentations (ie sub_seg, seg_bin_kidney_tumor_QC.nii.gz')", required=True )
	parser.add_argument ("--out_folder",  help="folder for csv to write to", required=True )
	parser.add_argument ("--out_filename",  help="filename for predictions", required=True )
	parser.add_argument ("--image_type",  help="CT or MR", required=True )
	parser.add_argument ("--add_bin",  help="1 or 0, add a comparison between binarized reference and binarized prediction", required=True )
	args = parser.parse_args()
	main(args)
