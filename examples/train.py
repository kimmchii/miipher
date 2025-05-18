import hydra
from lightning.pytorch.core import datamodule
from omegaconf import DictConfig
from lightning.pytorch import Trainer, seed_everything
import torch
from miipher import lightning_module
from miipher.lightning_module import MiipherLightningModule
from miipher.dataset.datamodule import MiipherDataModule


@hydra.main(version_base="1.3", config_name="config", config_path="./configs")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("medium")
    lightning_module = MiipherLightningModule(cfg)
    datamodule = MiipherDataModule(cfg)
    training_logger = hydra.utils.instantiate(cfg.train.loggers)
    trainer = hydra.utils.instantiate(cfg.train.trainer, logger=training_logger)
    trainer.fit(lightning_module, datamodule, ckpt_path=cfg.train.resume_from_checkpoint)


if __name__ == "__main__":
    main()
