# ============================================================================
# File: config.py
# Date: 2025-03-11
# Author: TA
# Description: Experiment configurations.
# ============================================================================

################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
exp_name = 'Semi_train'  # name of experiment

# Model Options
model_type = 'resnet18'  # 'mynet' or 'resnet18'

# Learning Options
epochs = 500                # train how many epochs
batch_size = 128            # batch size for dataloader 
use_adam = False           # Adam or SGD optimizer
lr = 1e-2                  # learning rate
milestones = [16, 32, 45]  # reduce learning rate at 'milestones' epochs