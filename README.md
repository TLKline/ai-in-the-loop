Compare submodel disagreement in nn-UNet

# Requirements
1. Install nnUNet per public github https://github.com/MIC-DKFZ/nnUNet*

2. Write out predictions on test set for final ensemble model and for EACH FOLD to separate folders (dir_ensemble_predictions, dir_fold1_predictions, .....)

* Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

# Steps
0. Write out test prediction vs. manual segmentation image-wise dice scores (step 0)

1. Calculate and summarize fold vs. fold image wise dice scores (step 1)

2. Compare test-manual and summarized fold-fold dice scores (step 2)
