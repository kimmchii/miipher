import hydra
from omegaconf import DictConfig
import torch
from miipher.lightning_module import MiipherLightningModule
from miipher.dataset.datamodule import MiipherDataModule
import argparse
import os.path as op

args = argparse.ArgumentParser()
args.add_argument(
    "--config",
    type=str,
    default="./configs/config.yaml",
    help="Path to the config file",
)
args = args.parse_args()

config_dir = op.dirname(args.config)
config_file_name = op.basename(args.config).split(".")[0]

@hydra.main(
    version_base="1.3",
    config_name=config_file_name,
    config_path=config_dir,
)
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("medium")
    lightning_module = MiipherLightningModule(cfg)
    datamodule = MiipherDataModule(cfg)
    training_logger = hydra.utils.instantiate(cfg.train.loggers)
    trainer = hydra.utils.instantiate(cfg.train.trainer, logger=training_logger)
    trainer.fit(lightning_module, datamodule, ckpt_path=cfg.train.resume_from_checkpoint)


if __name__ == "__main__":
    main()

