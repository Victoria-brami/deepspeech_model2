import os
import pytorch_lightning as pl
from pytorch_lightning import Callback
from torch.optim import Adam, AdamW, SGD
from optuna_model import DeepSpeech
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig, GCSCheckpointConfig, AdamConfig
from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from optuna_training_configuration import OptunaConfig, ExecutionConfig
from deepspeech_pytorch.lwlrap_loss import LWLRAP
import torch
import json
import torch.nn as nn

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import ray
import logging
import joblib

with open('labels.json', 'r') as jsn_file:
   labels = json.load(jsn_file)

cfg = DeepSpeechConfig()
optuna_config = OptunaConfig()
exec_config = ExecutionConfig()

def dump_study_callback(study, trial):
    joblib.dump(study, 'study.pkl')


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        print()
        print('Validation lwlrap: {}'.format(trainer.callback_metrics['val_lwlrap']))
        self.metrics.append(trainer.callback_metrics)


class LightningNet(pl.LightningModule):

    def __init__(self, trial):
        super(LightningNet, self).__init__()
        self.trial = trial

        data_loader = DeepSpeechDataModule(
            labels=labels,
            data_cfg=cfg.data,
            normalize=True,
            # is_distributed=cfg.trainer.gpus > 1
            is_distributed=False
        )

        model = DeepSpeech(
            labels=labels,
            model_cfg=cfg.model,
            optim_cfg=cfg.optim,
            precision=cfg.trainer.precision,
            spect_cfg=cfg.data.spect
        )
        self.model = model
        # self.optim_cfg = cfg.optim
        self.optim_cfg = AdamConfig()
        # self.model = model.cuda()

        # Choose loss
        if optuna_config.suggest_loss is not None:
            chosen_loss = self.trial.suggest_categorical('Loss', optuna_config.suggest_loss)
        else:
            chosen_loss = optuna_config.default_loss

        self.loss = nn.MSELoss()

        if chosen_loss == 'mseloss':
            self.loss = nn.MSELoss()
        elif chosen_loss == 'crossentropy':
            self.loss = nn.CrossEntropy()
        elif chosen_loss == 'bcewithlogitsloss':
            self.loss = nn.BCEWithLogitsLoss()

        self.lwlrap = LWLRAP(precision=16)

    def forward(self, inputs, inputs_sizes):
        return self.model(inputs, inputs_sizes)

    def training_step(self, batch, batch_idx):
        inputs, targets, input_percentages, target_sizes = batch
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        out, output_sizes = self(inputs, input_sizes)
        # print('TENSORS TYPES', out.type(), targets.type())
        train_loss = self.loss(out, targets.half())
        return train_loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, input_percentages, target_sizes = batch
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        out, output_sizes = self(inputs, input_sizes)
        val_lwlrap = self.lwlrap(out, targets)
        return {'batch_val_lwlrap': val_lwlrap}

    def validation_epoch_end(self, outputs):
        total_val_lwlrap = torch.stack([x['batch_val_lwlrap'] for x in outputs]).mean()
        self.log('val_lwlrap', total_val_lwlrap)
        return {'val_lwlrap': total_val_lwlrap}

    def configure_optimizers(self):
        # Learning rate suggestion
        if optuna_config.suggest_learning_rate is not None:  # choosing lr in the given interval
            chosen_lr = self.trial.suggest_loguniform('learning-rate',
                                                      optuna_config.suggest_learning_rate[0],
                                                      optuna_config.suggest_learning_rate[1])
        else:
            chosen_lr = optuna_config.default_learning_rate

        # Weight decay suggestion
        if optuna_config.suggest_weight_decay is not None:  # choosing wd in the given interval
            chosen_weight_decay = self.trial.suggest_loguniform('weight-decay',
                                                                optuna_config.suggest_weight_decay[0],
                                                                optuna_config.suggest_weight_decay[1])
        else:
            chosen_weight_decay = optuna_config.default_weight_decay
        
        print('Optimizer suggest', optuna_config.suggest_optimiser)
        
        # Optimiser suggestion
        if optuna_config.suggest_optimiser == 'SGD':
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=self.optim_cfg.learning_rate,
                momentum=self.optim_cfg.momentum,
                nesterov=True,
                weight_decay=self.optim_cfg.weight_decay
            )
        elif optuna_config.suggest_optimiser == 'Adam':
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.optim_cfg.learning_rate,
                betas=self.optim_cfg.betas,
                eps=self.optim_cfg.eps,
                weight_decay=self.optim_cfg.weight_decay
            )
        else:
            raise ValueError("Optimizer has not been specified correctly.")

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=self.optim_cfg.learning_anneal
        )
        return [optimizer], [scheduler]




def objective(trial):

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(exec_config.chkp_folder, "trial_{}".format(trial.number), "{epoch}"), monitor="val_lwlrap"
    )

    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(
        logger=False,
        checkpoint_callback=checkpoint_callback,
        max_epochs=exec_config.epochs,
        gpus=exec_config.gpus,
        callbacks=[metrics_callback],
        # callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_lwlrap")],
        # early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_lwlrap"),
        amp_level='O1',
        precision=16,
        num_sanity_val_steps=exec_config.num_validation_sanity_steps
    )
    data_loader = DeepSpeechDataModule(
        labels=labels,
        data_cfg=cfg.data,
        normalize=True,
        # is_distributed=cfg.trainer.gpus > 1
        is_distributed=False
    )

    model = LightningNet(trial)  # this initialisation depends on the trial argument
    trainer.fit(model, data_loader)

    print("METRIC ", metrics_callback.metrics)
    return metrics_callback.metrics[-1]["val_lwlrap"]


if __name__ == "__main__":

    if optuna_config.pruner == 'Hyperband':
        print('Hyperband pruner')
        pruner = optuna.pruners.HyperbandPruner(max_resource=optuna_config.n_iters,
                                                reduction_factor=optuna_config.reduction_factor)
    else:
        print('No pruner (or invalid pruner name)')
        pruner = optuna.pruners.NopPruner()

        # initialise the multiprocessing handler
    ray.init(num_cpus=4, num_gpus=exec_config.gpus, logging_level=logging.CRITICAL,
             ignore_reinit_error=True)

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=optuna_config.n_trials, timeout=optuna_config.timeout,
                   callbacks=[dump_study_callback], n_jobs=optuna_config.n_jobs)

    # displays a study summary
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # dumps the study for use with dash_study.py
    joblib.dump(study, 'study_finished.pkl')
