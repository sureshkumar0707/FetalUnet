# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 11:41:26 2018

@author: zfq
"""

import os

from medpy.io import load, save
import numpy as np

normal_dice = 0
normal_count = 0
ours_dice = 0
ours_count = 0
ft_dice = 0
ft_count = 0
inputCounter = 0
for f in os.walk('../InputData'):
    inputCounter += 1    
    if inputCounter > 1: 
        for f2 in os.listdir(f[0]): 
            if "Net_mask" in f2 and "nii" in f2 and "Normal" in f[0]:
                print(f2)
              
                predict_mask, image_header = load(f[0]+'/'+f2) # Load data
                gt_mask, gt_header = load(f[0] + '/' + 'ManualSegmentation/' + f2[4:])
                gt_mask = gt_mask.astype(np.float64)
                gt_mask[gt_mask == 0] = -1
                normal_dice = normal_dice + 2.0 * (np.sum(gt_mask == predict_mask))/(np.sum(predict_mask == 1) + np.sum(gt_mask == 1))                
                normal_count = normal_count + 1
           
            if "Net_mask" in f2 and "nii" in f2 and "SE" in f[0]:
                print(f2)
              
                predict_mask, image_header = load(f[0]+'/'+f2) # Load data
                gt_mask, gt_header = load(f[0] + '/' + 'gt_mask' + f[0][36:] + '.nii')
                gt_mask = gt_mask.astype(np.float64)
                gt_mask[gt_mask == 0] = -1
                ours_dice = ours_dice + 2.0 * (np.sum(gt_mask == predict_mask))/(np.sum(predict_mask == 1) + np.sum(gt_mask == 1))                
                ours_count = ours_count + 1
                
            if "ft_mask" in f2 and "nii" in f2 and "SE" in f[0]:
                print(f2)
              
                predict_mask, image_header = load(f[0]+'/'+f2) # Load data
                gt_mask, gt_header = load(f[0] + '/' + 'gt_mask' + f[0][36:] + '.nii')
                gt_mask = gt_mask.astype(np.float64)
                gt_mask[gt_mask == 0] = -1
                ft_dice = ft_dice + 2.0 * (np.sum(gt_mask == predict_mask))/(np.sum(predict_mask == 1) + np.sum(gt_mask == 1))                
                ft_count = ft_count + 1
                
                
if normal_count != 0:
    normal_dice = normal_dice/normal_count
if ours_count != 0:
    ours_dice = ours_dice/ours_count
if ft_count != 0:
    ft_dice = ft_dice/ft_count
print("normal_dice: ", normal_dice)
print("ours_dice: ", ours_dice)
print("ft_dice: ", ours_dice)

                


    


