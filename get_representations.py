import hydra
from hydra.core.config_store import ConfigStore
from hydra_configs.pytorch_lightning.callbacks import ModelCheckpointConf
import torch

from deepspeech_pytorch.configs.representations_config import DeepSpeechConfig, AdamConfig, SGDConfig, BiDirectionalConfig, \
    UniDirectionalConfig, GCSCheckpointConfig

import os
os.system('export PYTHONPATH=${PYTHONPATH}:$"/scratch2/vbrami/deepspeech_model2"')

from deepspeech_pytorch.extract_representations import representations_extractor

cs = ConfigStore.instance()
cs.store(name="config", node=DeepSpeechConfig)
cs.store(group="optim", name="sgd", node=SGDConfig)
cs.store(group="optim", name="adam", node=AdamConfig)
cs.store(group="checkpoint", name="file", node=ModelCheckpointConf)
cs.store(group="checkpoint", name="gcs", node=GCSCheckpointConfig)
cs.store(group="model", name="bidirectional", node=BiDirectionalConfig)
cs.store(group="model", name="unidirectional", node=UniDirectionalConfig)


@hydra.main(config_name="config")
def hydra_main(cfg):
    representations_extractor(cfg=cfg, layer='conv1', device='cuda')
    representations_extractor(cfg=cfg, layer='conv2', device='cuda')
    representations_extractor(cfg=cfg, layer='rnn1', device='cuda')



if __name__ == '__main__':
    hydra_main()