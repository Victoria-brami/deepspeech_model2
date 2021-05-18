import torch.nn as nn
import os
import numpy as np

class ExecutionConfig(object):
    """ Configuration of the training loop"""

    def __init__(self):
        self.epochs = 5 # 5
        self.gpus = 1
        self.num_validation_sanity_steps = 0

        os.makedirs('../optuna_training_weights/', exist_ok=True)
        self.chkp_folder = '../optuna_training_weights/'


class OptunaConfig(object):

    def __init__(self):
        n_iters = 10 # to be changed
        hours = 14
        reduction_factor = int(round(np.exp(np.log(n_iters) / 4)))

        self.n_jobs = 1  # number of parallel optimisations
        self.n_iters = n_iters
        self.timeout = hours * 3600
        self.reduction_factor = reduction_factor

        self.n_trials = 50  # 500          # will stop whenever the time or number of trials is reached
        self.pruner = 'Hyperband'  # options: Hyperband, Median, anything else -> no pruner

        # set default optimisr with Adam
        self.suggest_optimiser = 'Adam'  # default is hardcoded to Adam

        # Suggest loss
        self.suggest_loss = ['rmseloss', 'mseloss', 'crossentropy'] # None
        self.default_loss = 'mseloss'

        self.suggest_learning_rate = None  # [9e-4, 7e-3]
        self.default_learning_rate = 1.5e-4

        self.suggest_weight_decay = None  # [0, 1e-5]
        self.default_weight_decay = 1e-5
