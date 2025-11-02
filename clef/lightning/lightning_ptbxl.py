import os
import numpy as np
import torch
import lightning as L
from torch.utils.data import DataLoader
import torch.nn as nn
from clef.utils import ptb_utils as utils
import logging
from sklearn.metrics import f1_score
from typing import Optional

from clef.data.downstream_dataloader import PTBXLDataset
from clef.lightning.lightning_base import BaseModelClassifier

logger = logging.getLogger(__name__)


class PTBXL1LeadDataModule(L.LightningDataModule):
    def __init__(
        self,
        ecg_dir: str,
        task: str = "diagnostic",
        batch_size: int = 32,
        num_workers: int = 4,
        sampling_frequency: int = 500,
        min_samples: int = 0,
        output_folder: str = "./outputs/",
        input_len: int = 1000,
    ):
        super().__init__()
        self.ecg_dir = ecg_dir
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampling_frequency = sampling_frequency
        self.min_samples = min_samples
        self.output_folder = output_folder
        self.input_len = input_len
        self.experiment_name = f"ptbxl_{self.task}"
        self.lead_num = 1

    def setup(self, stage=None):
        self.data, self.raw_labels = utils.load_dataset(
            self.ecg_dir, self.sampling_frequency
        )
        self.labels = utils.compute_label_aggregations(
            self.raw_labels, self.ecg_dir, self.task
        )
        data_path = os.path.join(self.output_folder, self.experiment_name, "data")
        self.data, self.labels, self.Y, _ = utils.select_data(
            self.data, self.labels, self.task, self.min_samples, data_path
        )

        # Save data splits for evaluation
        tar_lead = 1
        x_test = self.data[self.labels.strat_fold == 10][:, :, [tar_lead]]
        y_test = self.Y[self.labels.strat_fold == 10]
        x_val = self.data[self.labels.strat_fold == 9][:, :, [tar_lead]]
        y_val = self.Y[self.labels.strat_fold == 9]
        x_train = self.data[self.labels.strat_fold <= 8][:, :, [tar_lead]]
        y_train = self.Y[self.labels.strat_fold <= 8]

        # Preprocess signals
        x_train, x_val, x_test = utils.preprocess_signals(
            x_train, x_val, x_test, data_path
        )
        self.mean_y = np.mean(y_train, axis=0)
        if stage == "fit" or stage is None:
            self.train_dataset = PTBXLDataset(
                x_train, y_train, input_len=self.input_len
            )
            self.val_dataset = PTBXLDataset(
                x_val, y_val, input_len=self.input_len
            )

            self.input_shape = x_train[0].shape
            self.n_classes = y_train.shape[1]

        if stage == "test" or stage is None:
            self.test_dataset = PTBXLDataset(
                x_test, y_test, input_len=self.input_len
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @property
    def num_classes(self):
        return self.n_classes


class PTBXL12LeadDataModule(L.LightningDataModule):
    def __init__(
        self,
        ecg_dir: str,
        task: str = "diagnostic",
        batch_size: int = 32,
        num_workers: int = 4,
        sampling_frequency: int = 100,
        min_samples: int = 0,
        output_folder: str = "./outputs/",
        input_len: int = 1000,
    ):
        super().__init__()
        self.ecg_dir = ecg_dir
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampling_frequency = sampling_frequency
        self.min_samples = min_samples
        self.output_folder = output_folder
        self.input_len = input_len
        self.experiment_name = f"ptbxl_{self.task}"
        self.lead_num = 12

    def setup(self, stage=None):
        self.data, self.raw_labels = utils.load_dataset(
            self.ecg_dir, self.sampling_frequency
        )
        self.labels = utils.compute_label_aggregations(
            self.raw_labels, self.ecg_dir, self.task
        )
        data_path = os.path.join(self.output_folder, self.experiment_name, "data")
        self.data, self.labels, self.Y, _ = utils.select_data(
            self.data, self.labels, self.task, self.min_samples, data_path
        )

        # Save data splits for evaluation
        x_test = self.data[self.labels.strat_fold == 10]
        y_test = self.Y[self.labels.strat_fold == 10]
        x_val = self.data[self.labels.strat_fold == 9]
        y_val = self.Y[self.labels.strat_fold == 9]
        x_train = self.data[self.labels.strat_fold <= 8]
        y_train = self.Y[self.labels.strat_fold <= 8]

        # Preprocess signals
        x_train, x_val, x_test = utils.preprocess_signals(
            x_train, x_val, x_test, data_path
        )
        self.mean_y = np.mean(y_train, axis=0)
        if stage == "fit" or stage is None:
            self.train_dataset = PTBXLDataset(
                x_train, y_train, input_len=self.input_len
            )
            self.val_dataset = PTBXLDataset(x_val, y_val, input_len=self.input_len)

            self.input_shape = x_train[0].shape
            self.n_classes = y_train.shape[1]

        if stage == "test" or stage is None:
            self.test_dataset = PTBXLDataset(x_test, y_test, input_len=self.input_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @property
    def num_classes(self):
        return self.n_classes


class PTBXLClassifier(BaseModelClassifier):
    def __init__(
        self,
        backbone,
        datamodule,
        num_classes: int,
        feature_dim: int = 768,
        do_proj: bool = False,
        thresholds: Optional[np.ndarray] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        optimizer: str = "adam",
        scheduler: Optional[str] = "onecycle",
        use_autocast: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "datamodule"])
        self.model = backbone
        self.num_classes = num_classes
        self.use_autocast = use_autocast
        self.thresholds = thresholds
        self.exp_name = f"ptbxl_{datamodule.task}_{backbone.name}_{datamodule.lead_num}lead"
        
        if do_proj:
            self.projection = nn.Linear(feature_dim, num_classes)
        else:
            self.projection = nn.Identity()

        self.criterion = nn.BCEWithLogitsLoss() # considered as multi-label classification

    def forward(self, x):
        if self.use_autocast and torch.cuda.is_available():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32,
            ):
                return self._forward_impl(x)
        else:
            return self._forward_impl(x)

    def _forward_impl(self, x):
        if self.model.__class__.__name__.lower() == "xresnet1d":
            feat = self.model(x)
            return self.projection(torch.mean(feat, dim=-1))
        elif self.model.__class__.__name__.lower() == 'st_mem_vit':
            x = x[:, :, :self.model.input_len]
            feat = self.model(x)
            return self.projection(feat)
        elif self.model.__class__.__name__.lower().startswith("moment"):
            feat = self.model(x)
            if hasattr(feat, 'embeddings'):
                return self.projection(feat.embeddings)
            else:
                return self.projection(feat)
        elif self.model.__class__.__name__.lower().startswith("moirai"):
            feat = self.capture_moirai_encoder_output(x)
            feat = torch.mean(feat, dim=1)
            return self.projection(feat)
        else:
            feat = self.model(x)
            return self.projection(feat)

    def optimize_thresholds(self, val_probs, val_targets):
        """Find optimal thresholds for each class based on F1 score"""
        best_thresholds = []
        for i in range(val_targets.shape[1]):
            best_f1 = 0
            best_thresh = 0.5
            for thresh in np.arange(0.1, 0.9, 0.05):
                preds = (val_probs[:, i] > thresh).astype(float)
                f1 = f1_score(val_targets[:, i], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            best_thresholds.append(best_thresh)
        return np.array(best_thresholds)

    def _is_binary_classification(self) -> bool:
        return True
    
    def _is_multitask_classification(self) -> bool:
        return True

