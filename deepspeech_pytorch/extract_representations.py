import torch
import json
from hydra.utils import to_absolute_path
from torch.cuda.amp import autocast
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf

from deepspeech_pytorch.model_for_extraction import DeepSpeech
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig, GCSCheckpointConfig, DataConfig
from deepspeech_pytorch.checkpoint import GCSCheckpointHandler, FileCheckpointHandler
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader, \
    DSElasticDistributedSampler, DSRandomSampler




def representations_extractor(layer: str,
                              checkpoint: str,
                              DEVICE: str,
                              is_distributed: bool,
                              data_cfg: DataConfig,
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
        audio_conf=data_cfg.spect,
        input_path=to_absolute_path(data_cfg.train_path),
        labels=labels,
        normalize=True,
        aug_cfg=data_cfg.augmentation
    )

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
        num_workers=data_cfg.num_workers,
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
    model.load_state_dict(torch.load(checkpoint))

    # Compute intermediate representations
    for i, batch in enumerate(loader, 0):

        with autocast(enabled=True):
            inputs, targets, input_percentages, target_sizes = batch
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            device = torch.cuda.device(DEVICE)
            inputs = inputs.to(device)
            out, output_sizes = model.intermediate_forward(inputs, input_sizes, layer)

            # Save the representations





    return None


if __name__ == '__main__':
    print()