# Methodology summary

Full details are in the accompanying manuscript. This is a short, evidence-based summary of what was executed.

* **Cohort.** 369 BraTS 2020 training directories; 368 converted. `BraTS20_Training_355` has a valid segmentation with a noncanonical filename, was not detected by the conversion step, and was therefore not converted, trained or evaluated. All results are n = 368.
* **Labels.** BraTS labels 0/1/2/4 were mapped to internal labels 0/2/1/3; evaluation regions are WT = {1,2,3}, TC = {2,3}, ET = {3}. Region-based (sigmoid) training with `regions_class_order = [1, 2, 3]`.
* **Model.** nnU-Net v2.8.1 `3d_fullres`, six-stage PlainConvUNet (features 32/64/128/256/320/320, InstanceNorm3d, LeakyReLU, deep supervision), 128³ patches, batch size 2.
* **Training.** 1,000 epochs × 250 iterations, SGD with Nesterov momentum 0.99, polynomial learning-rate decay, Dice + BCE loss.
* **Validation protocol.** Five patient-level folds (74/74/74/73/73 held-out). The held-out fold was used for epoch-wise patch-level monitoring and for nnU-Net's `checkpoint_best` selection; reported predictions were generated from `checkpoint_final`, identified from logs, timestamps and source code (command lines for folds 1–4 were not archived).
* **Inference.** Sliding window 128³, tile step 0.5, Gaussian weighting, mirroring on all axes, sigmoid threshold 0.5, hierarchical label map, no postprocessing.
* **Metrics.** Dice, IoU, sensitivity, precision, specificity, HD95 (pooled bidirectional percentile, mm), absolute relative volume error. Empty-region policy and statistics: see README and `src/brats_eval/summarize.py`.
* **Limitations.** Internal cross-validation only; no external validation, comparator or clinical evaluation.
