import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import lightning as L
from collections import Counter
from typing import Optional


from clef.data.downstream_dataloader import IcentiaBeatDataset, IcentiaRhythmDataset
from clef.lightning.lightning_base import BaseModelClassifier
from clef.utils.utils import load_records


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss


class BeatDataModule(L.LightningDataModule):
    def __init__(
        self,
        ecg_dir,
        batch_size=32,
        num_workers=4,
        input_len=1000,
        train_split=0.7,
        val_split=0.15,
        use_weighted_sampler=True,
    ):
        super().__init__()
        self.ecg_dir = ecg_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_len = input_len
        self.train_split = train_split
        self.val_split = val_split
        self.use_weighted_sampler = use_weighted_sampler
        self.lead_num = 1
        self.task = "beat"

    def prepare_data(self):
        self.all_patients = list(load_records(os.path.join(self.ecg_dir, "RECORDS")))

    def setup(self, stage=None):
        # This method is called on every GPU
        if not hasattr(self, 'all_patients'):
            self.prepare_data()
        
        # Split patients into train/val/test
        np.random.shuffle(self.all_patients)
        
        n_patients = len(self.all_patients)
        train_size = int(n_patients * self.train_split)
        val_size = int(n_patients * self.val_split)
        
        train_patients = self.all_patients[:train_size]
        val_patients = self.all_patients[train_size:train_size + val_size]
        test_patients = self.all_patients[train_size + val_size:]
        
        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = IcentiaBeatDataset(
                self.ecg_dir, train_patients, "train", self.input_len
            )
            self.val_dataset = IcentiaBeatDataset(
                self.ecg_dir, val_patients, "val", self.input_len
            )
            
            # Create weighted sampler for training if needed
            if self.use_weighted_sampler:
                self.train_sampler = self._create_weighted_sampler(self.train_dataset)
            else:
                self.train_sampler = None
                
            # Store number of classes
            self.n_classes = len(self.train_dataset.beat_types)

        if stage == "test" or stage is None:
            self.test_dataset = IcentiaBeatDataset(
                self.ecg_dir, test_patients, "test", self.input_len
            )

    def _create_weighted_sampler(self, dataset):
        """Create a weighted sampler to handle class imbalance"""
        labels = [dataset.samples[i]['label'] for i in range(len(dataset))]
        
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        print(f"Training set class distribution: {class_counts}")
        
        # Calculate weights inversely proportional to class frequency
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in labels]
        
        print(f"Sample weights per class: {class_weights}")
        
        return WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler if self.use_weighted_sampler else None,
            shuffle=not self.use_weighted_sampler,
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


class RhythmDataModule(L.LightningDataModule):
    def __init__(
        self,
        ecg_dir,
        batch_size=32,
        num_workers=4,
        input_len=1000,
        train_split=0.7,
        val_split=0.15,
        use_weighted_sampler=True,
    ):
        super().__init__()
        self.ecg_dir = ecg_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_len = input_len
        self.train_split = train_split
        self.val_split = val_split
        self.use_weighted_sampler = use_weighted_sampler
        self.lead_num = 1
        self.task = "rhythm"

    def prepare_data(self):
        self.all_patients = list(load_records(os.path.join(self.ecg_dir, "RECORDS")))

    def setup(self, stage=None):
        # This method is called on every GPU
        if not hasattr(self, 'all_patients'):
            self.prepare_data()
        
        # Split patients into train/val/test
        np.random.shuffle(self.all_patients)
        
        n_patients = len(self.all_patients)
        train_size = int(n_patients * self.train_split)
        val_size = int(n_patients * self.val_split)
        
        train_patients = self.all_patients[:train_size]
        val_patients = self.all_patients[train_size:train_size + val_size]
        test_patients = self.all_patients[train_size + val_size:]
        
        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = IcentiaRhythmDataset(
                self.ecg_dir, train_patients, "train", self.input_len
            )
            self.val_dataset = IcentiaRhythmDataset(
                self.ecg_dir, val_patients, "val", self.input_len
            )
            
            # Create weighted sampler for training if needed
            if self.use_weighted_sampler:
                self.train_sampler = self._create_weighted_sampler(self.train_dataset)
            else:
                self.train_sampler = None
                
            # Store number of classes
            self.n_classes = len(self.train_dataset.rhythm_types)

        if stage == "test" or stage is None:
            self.test_dataset = IcentiaRhythmDataset(
                self.ecg_dir, test_patients, "test", self.input_len
            )

    def _create_weighted_sampler(self, dataset):
        """Create a weighted sampler to handle class imbalance"""
        labels = [dataset.samples[i]['label'] for i in range(len(dataset))]
        
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        print(f"Training set class distribution: {class_counts}")
        
        # Calculate weights inversely proportional to class frequency
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in labels]
        
        print(f"Sample weights per class: {class_weights}")
        
        return WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler if self.use_weighted_sampler else None,
            shuffle=not self.use_weighted_sampler,
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
    

class IcentiaClassifier(BaseModelClassifier):
    def __init__(
        self,
        backbone,
        datamodule,
        do_proj=False,
        num_classes=3, 
        lr=1e-4,
        weight_decay=1e-5,
        optimizer: str = "adam",
        scheduler: Optional[str] = "cosinewarmup",
        feature_dim=768,
        use_focal_loss=False,
        use_weighted_loss=False,
        use_autocast=True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "datamodule"])
        self.model = backbone
        self.num_classes = num_classes
        self.use_autocast = use_autocast
        self.exp_name = f"icentia11k_{datamodule.task}_{backbone.name}_{datamodule.lead_num}lead"
        
        if do_proj:
            self.projection = nn.Linear(feature_dim, num_classes)
        else:
            self.projection = nn.Identity()

        if use_weighted_loss and hasattr(datamodule, 'class_weight'):
            sorted_weights = [datamodule.class_weight.get(i, 1.0) for i in range(num_classes)]
            weight_tensor = torch.tensor(sorted_weights, dtype=torch.float)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        elif use_focal_loss:
            self.criterion = FocalLoss(alpha=1, gamma=2)
        else:
            self.criterion = nn.CrossEntropyLoss()

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
        return False
    
    def _is_multitask_classification(self) -> bool:
        return False
   
