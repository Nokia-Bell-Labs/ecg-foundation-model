import os
import inspect
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
from clef.task_config import TaskConfig
from clef.utils.utils import set_seed

from omegaconf import DictConfig, OmegaConf
import hydra

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="downstream", version_base=None)
def run_downstream(cfg: DictConfig):
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

    ##### Task Configuration #####
    task_config = TaskConfig(
        name=cfg.task.name, task=cfg.exp.get("task", None), lead_config=cfg.task.get("lead_config", "1lead")
    )

    ##### Data Configuration #####
    data_exp_params = (
        OmegaConf.to_container(cfg.task.params, resolve=True)
        if "params" in cfg.task
        else {}
    )

    # For datasets with multiple tasks, set specific task
    data_exp_params["task_type"] = cfg.task.type
    if cfg.task.name in ["ptbxl", "aurorabp", "mcmed"] and hasattr(cfg.exp, "task"):
        data_exp_params["task"] = cfg.exp.task

    # Filter parameters to only include those accepted by the datamodule
    datamodule_class = task_config.datamodule_class
    accepted_params = inspect.signature(datamodule_class.__init__).parameters.keys()
    filtered_params = {k: v for k, v in data_exp_params.items() if k in accepted_params}

    # Load data
    datamodule = task_config.datamodule_class(**filtered_params)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    ##### Model Configuration #####
    model_register = ModelConfig(name=cfg.model.name)
    model_params = OmegaConf.create(
        {
            **OmegaConf.to_container(cfg.model.model_params, resolve=True),
            "sample_rate": cfg.exp.sample_rate,
            "input_len": cfg.exp.input_len,
            "feature_dim": cfg.model.model_params.feature_dim,
            "lead_config": cfg.task.get("lead_config", "12lead"),
        }
    )

    task_model_params = {}
    if "model_params" in cfg.task:
        task_model_params = OmegaConf.to_container(cfg.task.model_params, resolve=True)

    def _set_num_classes(task_model_params, model_params, num_classes):
        """Helper function to set num_classes in both parameter dictionaries."""
        task_model_params["num_classes"] = num_classes
        model_params["num_classes"] = num_classes

    if cfg.task.name == "ptbxl":
        if "num_classes" not in task_model_params:
            num_classes = task_config.get_ptbxl_num_classes()
            _set_num_classes(task_model_params, model_params, num_classes)
    elif cfg.task.name == "mcmed":
        if cfg.task.type == "regression":
            _set_num_classes(task_model_params, model_params, 1)
        else:
            num_classes = task_config.get_mcmed_num_classes()
            _set_num_classes(task_model_params, model_params, num_classes)
    elif cfg.task.name == "mimiciv":
        if cfg.task.type == "regression":
            _set_num_classes(task_model_params, model_params, 1)
        else:
            _set_num_classes(task_model_params, model_params, cfg.exp.num_classes)
    elif cfg.task.name in ["icentia", "chapman", "music"]:
        # Datasets that only support classification
        _set_num_classes(task_model_params, model_params, cfg.exp.num_classes)
    elif cfg.task.name == "aurorabp":
        # Dataset that only supports regression
        _set_num_classes(task_model_params, model_params, 1)
    else:
        raise ValueError(
            f"Unknown task: {cfg.task.name}. Available tasks: {list(task_config.datamodule_class.__name__)}"
        )

    if "checkpoint_path" in cfg.model:
        model_params["checkpoint_path"] = cfg.model.checkpoint_path
    if "statekey_file" in cfg.model:
        model_params["statekey_file"] = cfg.model.statekey_file

    # Initialize model
    backbone = model_register.model_class(model_params)
    task_model_class = task_config.get_model_class(cfg.task.type)
    model = task_model_class(
        backbone=backbone, datamodule=datamodule, **task_model_params
    )
    logger.info(
        f"Downstream evaluation for model {cfg.model.name} on task {cfg.task.name}"
    )

    ##### Trainer Configuration #####
    trainer = L.Trainer(
        max_epochs=cfg.exp.epochs,
        callbacks=[progress_bar, checkpoint_callback, early_stop_callback],
        accelerator=cfg.exp.accelerator,
        devices=[cfg.exp.devices],
        precision="16-mixed",
        logger=False,
    )

    if cfg.exp.finetune:
        trainer.fit(model, train_loader, val_loader)
    if test_loader is not None:
        trainer.test(model, test_loader, ckpt_path=checkpoint_callback.best_model_path)

    final_model_path = os.path.join(output_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    run_downstream()
