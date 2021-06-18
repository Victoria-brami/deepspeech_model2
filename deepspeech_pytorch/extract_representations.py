import torch
import json
import os
import numpy as np
from hydra.utils import to_absolute_path
from torch.cuda.amp import autocast
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf

from deepspeech_pytorch.model_for_extraction import DeepSpeech
from deepspeech_pytorch.configs.representations_config import DeepSpeechConfig, GCSCheckpointConfig, DataConfig
from deepspeech_pytorch.checkpoint import GCSCheckpointHandler, FileCheckpointHandler
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader, \
    DSElasticDistributedSampler, DSRandomSampler

"""
def reconcatenate_image(image_output):
    
    # image output of size [1, nb_channels, H, W]
    nb_channels = image_output.shape[1]
    W = image_output.shape[3]
    H = image_output.shape[2]
    
    new_image = np.zeros((H, W * nb_channels))
    
    for i in range(nb_channels):
        X = image_output[0, i, :, :].cpu().detach().numpy()
        first_im = X
        new_image[:, i*W:(i+1)*W] = first_im
    
    return new_image"""

def reshape_outputs(outputs, layer):
    # W : time dimension

    if layer.startswith('input'):
        # image output of size [1, nb_channels, H, W]
        return outputs[0, 0, :, :]

    elif layer.startswith('conv'):
        # image output of size [1, nb_channels, H, W]
        nb_channels = outputs.shape[1]
        W = outputs.shape[3]
        H = outputs.shape[2]
        reshaped_output = np.zeros((H * nb_channels, W ))
        for i in range(nb_channels):
            X = outputs[0, i, :, :].cpu().detach().numpy()
            first_im = X
            reshaped_output[i * H:(i + 1) * H, :] = first_im

    elif layer.startswith('rnn'):
        # image output of size [W, 1, H]
        reshaped_output = outputs[:, 0, :].cpu().detach().numpy()
        reshaped_output = reshaped_output.transpose()
    elif layer.startswith('lookahead'):
        # image output of size [W, 1, H]
        W = outputs.shape[0]
        H = outputs.shape[2]
        reshaped_output = outputs[:, 0, :].cpu().detach().numpy()
        reshaped_output = reshaped_output.transpose()
    elif layer.startswith('fully_connected'):
        # image output of size [W, 1, H]
        reshaped_output = outputs[:, 0, :].cpu().detach().numpy()
        reshaped_output = reshaped_output.transpose()
    return reshaped_output


def save_outputs(outputs, layer, i, destination_path, dataset_name):

    full_destination_path = os.path.join(destination_path, layer, dataset_name)
    os.makedirs(full_destination_path, exist_ok=True)
    save_filename = '{}.npy'.format(i)
    np.save(os.path.join(full_destination_path, save_filename), outputs)


def representations_extractor(layer: str,
                              # dataset_name: str,
                              # destination_path: str,
                              # checkpoint: str,
                              device: str,
                              cfg: DeepSpeechConfig):
    seed_everything(cfg.seed)

    # Load the Labels (USELESS UP TO NOW)
    with open(to_absolute_path(cfg.data.labels_path)) as label_file:
        labels = json.load(label_file)
    print('Loaded Labels', labels)

    # Load the checkpoint
    if cfg.trainer.checkpoint_callback:
        if OmegaConf.get_type(cfg.checkpoint) is GCSCheckpointConfig:
            checkpoint_callback = GCSCheckpointHandler(
                cfg=cfg.checkpoint
            )
        else:
            checkpoint_callback = FileCheckpointHandler(
                cfg=cfg.checkpoint
            )
        if cfg.load_auto_checkpoint:
            resume_from_checkpoint = checkpoint_callback.find_latest_checkpoint()
            if resume_from_checkpoint:
                cfg.trainer.resume_from_checkpoint = resume_from_checkpoint


    # Define the dataloader
    print('Load the Dataset ...')
    dataset = SpectrogramDataset(
        audio_conf=cfg.data.spect,
        input_path=to_absolute_path(cfg.data.train_path),
        labels=labels,
        normalize=True,
        aug_cfg=cfg.data.augmentation
    )
    is_distributed = False
    if is_distributed:
        sampler = DSElasticDistributedSampler(
            dataset=dataset,
            batch_size=1
        )
    else:
        sampler = DSRandomSampler(
            dataset=dataset,
            batch_size=1
        )
    loader = AudioDataLoader(
        dataset=dataset,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        batch_sampler=sampler
    )

    # Load the model
    model = DeepSpeech(
        labels=labels,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        precision=cfg.trainer.precision,
        spect_cfg=cfg.data.spect
    )
    # model.load_state_dict(torch.load(checkpoint))

    # Compute intermediate representations
    for i, batch in enumerate(loader, 0):

        with autocast(enabled=True):
            inputs, targets, input_percentages, target_sizes = batch
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            # device = torch.cuda.device(device)
            # inputs = inputs.to(device)
            out, output_sizes = model.intermediate_forward(inputs, input_sizes, layer)

            if i == 0:
                print('Layer {} output shape: {}'.format(layer, out.shape))

            # Reshape the outputs
            new_outputs = reshape_outputs(out, layer)

            # Save the representations
            dataset_name = 'freesound_train_curated'
            destination_path = 'freesound_outputs'
            save_outputs(out, layer, i, destination_path, dataset_name)

    return None


# def main_deepspeech_extraction(args):

