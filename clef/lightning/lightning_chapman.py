import torch
import lightning as L
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
from typing import Optional

from clef.data.downstream_dataloader import Chapman1LeadDataset, Chapman12LeadDataset
from clef.lightning.lightning_base import BaseModelClassifier

logger = logging.getLogger(__name__)


class Chapman1LeadDataModule(L.LightningDataModule):
    def __init__(
        self,
        ecg_dir,
        batch_size=64,
        num_workers=4,
        task_type="classification",
        input_len=1000,
        target_fs=100,
        normalize=True,
        lead_idx=1, # Default to lead II
    ):
        super().__init__()
        self.data_path = ecg_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_type = task_type
        self.input_len = input_len
        self.target_fs = target_fs
        self.normalize = normalize
        self.lead_idx = lead_idx
        self.lead_num = 1

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Chapman1LeadDataset(
                data_path=self.data_path,
                stage="train",
                task=self.task_type,
                input_len=self.input_len,
                target_fs=self.target_fs,
                normalize=self.normalize,
                lead_idx=self.lead_idx,
            )
            self.val_dataset = Chapman1LeadDataset(
                data_path=self.data_path,
                stage="val",
                task=self.task_type,
                input_len=self.input_len,
                target_fs=self.target_fs,
                normalize=self.normalize,
                lead_idx=self.lead_idx,
            )

        if stage == "test" or stage is None:
            self.test_dataset = Chapman1LeadDataset(
                data_path=self.data_path,
                stage="test",
                task=self.task_type,
                input_len=self.input_len,
                target_fs=self.target_fs,
                normalize=self.normalize,
                lead_idx=self.lead_idx,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def get_class_weights(self):
        """Return class weights from the training dataset"""
        if hasattr(self, 'train_dataset'):
            return self.train_dataset.get_class_weights()
        return None
    
    def get_num_classes(self):
        """Return number of classes from the training dataset"""
        if hasattr(self, 'train_dataset'):
            return self.train_dataset.labels.shape[1]
        return None
    

class Chapman12LeadDataModule(L.LightningDataModule):
    def __init__(
        self,
        ecg_dir,
        batch_size=64,
        num_workers=4,
        task_type="classification",
        input_len=1000,
        target_fs=100,
        normalize=True,
    ):
        super().__init__()
        self.data_path = ecg_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_type = task_type
        self.input_len = input_len
        self.target_fs = target_fs
        self.normalize = normalize
        self.lead_num = 12

    def setup(self, stage=None):
        """Set up the datasets for each stage"""
        if stage == "fit" or stage is None:
            self.train_dataset = Chapman12LeadDataset(
                data_path=self.data_path,
                stage="train",
                task=self.task_type,
                input_len=self.input_len,
                target_fs=self.target_fs,
                normalize=self.normalize,
            )
            self.val_dataset = Chapman12LeadDataset(
                data_path=self.data_path,
                stage="val",
                task=self.task_type,
                input_len=self.input_len,
                target_fs=self.target_fs,
                normalize=self.normalize,
            )

        if stage == "test" or stage is None:
            self.test_dataset = Chapman12LeadDataset(
                data_path=self.data_path,
                stage="test",
                task=self.task_type,
                input_len=self.input_len,
                target_fs=self.target_fs,
                normalize=self.normalize,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def get_class_weights(self):
        """Return class weights from the training dataset"""
        if hasattr(self, 'train_dataset'):
            return self.train_dataset.get_class_weights()
        return None
    
    def get_num_classes(self):
        """Return number of classes from the training dataset"""
        if hasattr(self, 'train_dataset'):
            return self.train_dataset.labels.shape[1]
        return None


class ChapmanClassifier(BaseModelClassifier):
    def __init__(
        self,
        backbone,
        datamodule,
        do_proj=False,
        num_classes=4,  # Chapman has 4 rhythm classes typically
        lr=1e-4,
        weight_decay=1e-5,
        optimizer: str = "adam",
        scheduler: Optional[str] = "plateau",
        feature_dim=768,
        use_autocast=True,
        class_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "datamodule"])
        self.model = backbone
        self.num_classes = num_classes
        self.use_autocast = use_autocast
        self.num_classes = num_classes
        self.exp_name = f"chapman_{datamodule.task_type}_{backbone.name}_{datamodule.lead_num}lead"

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

    def _is_binary_classification(self) -> bool:
        return True
    
    def _is_multitask_classification(self) -> bool:
        return True



