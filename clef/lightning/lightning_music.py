import os

import pandas as pd
import lightning as L
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from typing import Optional
from collections import Counter

from clef.data.downstream_dataloader import MusicDataset
from clef.lightning.lightning_base import BaseModelClassifier

    
class MusicDataModule(L.LightningDataModule):
    def __init__(
        self,
        ecg_dir,
        label_csv,
        batch_size=64,
        num_workers=4,
        task_type="classification",
        input_len=1000,
        use_weighted_sampler=True,
    ):
        super().__init__()
        self.ecg_dir = ecg_dir
        self.label_csv = label_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_type = task_type
        self.input_len = input_len
        self.use_weighted_sampler = use_weighted_sampler
        self.lead_num = 1

    def setup(self, stage=None):
        if (
            os.path.exists("dataset/music/train_music.csv")
            and os.path.exists("dataset/music/val_music.csv")
            and os.path.exists("dataset/music/test_music.csv")
        ):
            self.train_dataset = MusicDataset(
                self.ecg_dir,
                "dataset/music/train_music.csv",
                self.task_type,
                input_len=self.input_len,
            )
            self.val_dataset = MusicDataset(
                self.ecg_dir,
                "dataset/music/val_music.csv",
                self.task_type,
                input_len=self.input_len,
            )
            self.test_dataset = MusicDataset(
                self.ecg_dir,
                "dataset/music/test_music.csv",
                self.task_type,
                input_len=self.input_len,
            )

        else:
            df = pd.read_csv(self.label_csv)
            # Exclude missing files
            with open("dataset/music/music_missing_files.txt") as f:
                missing = set(line.strip() for line in f if line.strip())
            df = df[~df["subj_id"].isin(missing)].reset_index(drop=True)

            if self.task_type == "classification":
                label_map = {0: 0, 1: 1, 3: 2, 6: 3, 7: 3}
                df = df[df["outcome"].isin(label_map.keys())].reset_index(drop=True)
                df["outcome"] = df["outcome"].map(label_map)
            else:
                raise ValueError("MUSIC dataset only support classification tasks.")

            train = df.iloc[: int(0.7 * len(df))]
            val = df.iloc[int(0.7 * len(df)) : int(0.8 * len(df))]
            test = df.iloc[int(0.8 * len(df)) :]

            train.to_csv("dataset/music/train_music.csv", index=False)
            val.to_csv("dataset/music/val_music.csv", index=False)
            test.to_csv("dataset/music/test_music.csv", index=False)
            self.train_dataset = MusicDataset(
                self.ecg_dir, "dataset/music/train_music.csv", self.task_type
            )
            self.val_dataset = MusicDataset(
                self.ecg_dir, "dataset/music/val_music.csv", self.task_type
            )
            self.test_dataset = MusicDataset(
                self.ecg_dir, "dataset/music/test_music.csv", self.task_type
            )

        if self.task_type == "classification":
            self.class_weight = self.train_dataset.class_weight

            if self.use_weighted_sampler:
                self.train_sampler = self._create_weighted_sampler(self.train_dataset)
            else:
                self.train_sampler = None
        
        else:
            raise ValueError("MUSIC dataset only support classification tasks.")

    def _create_weighted_sampler(self, dataset):
        """Create a weighted sampler to handle class imbalance"""
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label.item())
        
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        print(f"Training set class distribution: {class_counts}")
        
        # Calculate weights inversely proportional to class frequency
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in labels]
        
        print(f"Sample weights per class: {class_weights}")
        
        return WeightedRandomSampler(
            weights=sample_weights,
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


class MusicClassifier(BaseModelClassifier):
    def __init__(
        self,
        backbone,
        datamodule,
        do_proj=False,
        num_classes=4,
        lr=1e-4,
        weight_decay=1e-5,
        optimizer: str = "adam",
        scheduler: Optional[str] = "cosinewarmup",
        feature_dim=768,
        use_autocast=True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.model = backbone
        self.num_classes = num_classes
        self.use_autocast = use_autocast
        self.exp_name = f"music_{datamodule.task_type}_{backbone.name}_{datamodule.lead_num}lead"

        if do_proj:
            self.projection = nn.Linear(feature_dim, num_classes)
        else:
            self.projection = nn.Identity()

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
        """Music classification is multiclass (4 classes), so return False"""
        return False

    def _is_multitask_classification(self) -> bool:
        return False
