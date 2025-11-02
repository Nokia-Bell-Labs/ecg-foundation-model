import os
import warnings
import logging
import lightning as L

from lightning.pytorch.callbacks import (
    RichProgressBar,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from clef.model_config import ModelConfig
from clef.data_config import DataConfig
from clef.utils.utils import set_seed
from clef.lightning.lightning_contrastive import ContrastiveModel

from omegaconf import DictConfig, OmegaConf
import hydra

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="pretrain", version_base=None)
def run_pretrain(cfg: DictConfig):
    os.environ["HYDRA_FULL_ERROR"] = "1"
    set_seed(cfg.seed)

    ##### Callbacks #####
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="#08445C",
            progress_bar="#FF00FF",
            progress_bar_finished="#27040C",
            progress_bar_pulse="#6206E0",
            batch_progress="#FAD609",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter="\n",
            metrics_format=".3e",
        )
    )

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg["runtime"]["output_dir"]
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=cfg.exp.ckpt_name,
        save_top_k=1,
        monitor=cfg.exp.monitor_metric,
    )

    early_stop_callback = EarlyStopping(
        monitor=cfg.exp.monitor_metric,
        patience=cfg.exp.patience,
        verbose=True,
        mode=cfg.exp.patience_mode,
        min_delta=cfg.exp.patience_delta,
    )

    ##### Data Configuration #####
    data_exp_params = {
        **OmegaConf.to_container(cfg.exp, resolve=True),
    }

    # Load data
    data_register = DataConfig(
        name=cfg.dataset.data_group,
    )
    dataset = data_register.data_module_class(
        data_path=cfg.dataset.data_path,
        metadata_path=cfg.dataset.metadata_path,
        **data_exp_params,
    )
    dataset.setup()
    train_loader = dataset.train_dataloader()

    ##### Model Configuration #####
    model_register = ModelConfig(
        name=cfg.model.name,
    )
    model_params = OmegaConf.create(
        {
            **OmegaConf.to_container(cfg.model.model_params, resolve=True),
            "sample_rate": cfg.exp.sample_rate,
            "input_len": cfg.exp.input_len,
            "output_dim": cfg.exp.output_dim,
        }
    )

    ##### Experiment Configuration #####
    exp_params = OmegaConf.create({**OmegaConf.to_container(cfg.exp, resolve=True)})
    backbone = model_register.model_class(
        model_params, datamodule=dataset, **exp_params
    )
    model = ContrastiveModel(backbone, model_params, exp_params)

    logger.info(f"Pretraining model: {cfg.model.name}")
    trainer = L.Trainer(
        max_epochs=cfg.exp.epochs,
        callbacks=[progress_bar, checkpoint_callback, early_stop_callback],
        accelerator=cfg.exp.accelerator,
        devices=[cfg.exp.devices],
        logger=False,
    )
    trainer.fit(model, train_loader)

    if cfg.exp.pretrain_method == "simclr":
        pretrain_info = "meta" if cfg.exp.use_metadata else "simclr"
    else:
        pretrain_info = cfg.exp.pretrain_method

    param_str = (
        f"{cfg.model.name}-{cfg.model.model_size}_ep{cfg.exp.epochs}_{pretrain_info}"
    )
    deterministic_ckpt = f"./models/{param_str}_final_model.ckpt"
    trainer.save_checkpoint(deterministic_ckpt)
    logger.info(f"Deterministic checkpoint saved to {deterministic_ckpt}")


if __name__ == "__main__":
    run_pretrain()
