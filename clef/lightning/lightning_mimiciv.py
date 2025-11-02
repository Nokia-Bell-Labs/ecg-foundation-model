import torch
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Optional
from sklearn.preprocessing import StandardScaler

from clef.data.downstream_dataloader import (
    LVEF_1lead_cls_Dataset,
    LVEF_1lead_reg_Dataset,
    LVEF_12lead_cls_Dataset,
    LVEF_12lead_reg_Dataset,
)
from clef.lightning.lightning_base import BaseModelClassifier, BaseModelRegressor


class LVEF1LeadDataModule(L.LightningDataModule):
    def __init__(
        self,
        ecg_dir,
        label_csv,
        batch_size=64,
        num_workers=4,
        task_type="classification",
        input_len=5000,
    ):
        super().__init__()
        self.ecg_dir = ecg_dir
        self.label_csv = label_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_type = task_type
        self.lead_num = 1

    def setup(self, stage=None):
        df = pd.read_csv(self.label_csv)
        train, test = df.iloc[: int(0.8 * len(df))], df.iloc[int(0.8 * len(df)) :]
        val, test = test.iloc[: len(test) // 2], test.iloc[len(test) // 2 :]
        
        if self.task_type == "classification":
            DatasetClass = LVEF_1lead_cls_Dataset

            # No scaling for classification
            self.scalar = None
        else:
            DatasetClass = LVEF_1lead_reg_Dataset

            self.scaler = StandardScaler()
            train["LVEF"] = self.scaler.fit_transform(train[["LVEF"]])
            val["LVEF"] = self.scaler.transform(val[["LVEF"]])
            test["LVEF"] = self.scaler.transform(test[["LVEF"]])

        self.train_dataset = DatasetClass(self.ecg_dir, train)
        self.val_dataset = DatasetClass(self.ecg_dir, val)
        self.test_dataset = DatasetClass(self.ecg_dir, test)

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

    def inverse_transform_targets(self, targets):
        """Inverse transform targets back to original scale"""
        if self.scaler is not None and self.task_type == "regression":
            if isinstance(targets, torch.Tensor):
                targets_np = targets.detach().cpu().numpy()
            else:
                targets_np = targets
            
            return self.scaler.inverse_transform(targets_np.reshape(-1, 1)).flatten()
        return targets


class LVEF12LeadDataModule(L.LightningDataModule):
    def __init__(
        self, ecg_dir, label_csv, batch_size=64, num_workers=4, task_type="classification"
    ):
        super().__init__()
        self.ecg_dir = ecg_dir
        self.label_csv = label_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_type = task_type
        self.lead_num = 12

    def setup(self, stage=None):
        df = pd.read_csv(self.label_csv)
        train, test = df.iloc[: int(0.8 * len(df))], df.iloc[int(0.8 * len(df)) :]
        val, test = test.iloc[: len(test) // 2], test.iloc[len(test) // 2 :]
        if self.task_type == "classification":
            DatasetClass = LVEF_12lead_cls_Dataset
        else:
            DatasetClass = LVEF_12lead_reg_Dataset
        self.train_dataset = DatasetClass(self.ecg_dir, train)
        self.val_dataset = DatasetClass(self.ecg_dir, val)
        self.test_dataset = DatasetClass(self.ecg_dir, test)

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


class LVEFClassifier(BaseModelClassifier):
    def __init__(
        self,
        backbone,
        datamodule,
        do_proj=False,
        num_classes=2,
        lr=1e-4,
        weight_decay=1e-5,
        optimizer: str = "adam",
        scheduler: Optional[str] = "plateau",
        feature_dim=768,
        use_autocast=True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.model = backbone
        self.num_classes = num_classes
        self.use_autocast = use_autocast
        self.exp_name = f"mimiciv_{datamodule.task_type}_{backbone.name}_{datamodule.lead_num}lead"

        if do_proj:
            self.projection = nn.Linear(feature_dim, 1)
        else:
            self.projection = nn.Identity()

        self.criterion = nn.BCEWithLogitsLoss()

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
        """MIMICIV is binary classification, so return True"""
        return True

    def _is_multitask_classification(self) -> bool:
        return False
    
    
class LVEFRegressor(BaseModelRegressor):
    def __init__(
        self,
        backbone,
        datamodule,
        do_proj=False,
        lr=1e-4,
        weight_decay=1e-5,
        optimizer: str = "adam",
        scheduler: Optional[str] = "cosinewarmup",
        feature_dim=768,
        use_autocast=True,
        num_classes=1,
    ):
        super().__init__(datamodule=datamodule)
        self.save_hyperparameters(ignore=["backbone", "datamodule"])
        self.model = backbone
        self.use_autocast = use_autocast
        self.exp_name = f"mimiciv_{datamodule.task_type}_{backbone.name}_{datamodule.lead_num}lead"
        
        # Create projection layer if needed
        if do_proj:
            self.projection = nn.Linear(feature_dim, num_classes)
        else:
            self.projection = nn.Identity()
            
        self.criterion = nn.L1Loss()

    def forward(self, x):
        """Forward pass through the model and projection layer."""
        if self.model.__class__.__name__.lower() == "xresnet1d":
            x = x[:, :, :1000]
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
        
