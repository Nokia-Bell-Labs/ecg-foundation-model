import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import lightning as L
from collections import Counter
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from clef.data.downstream_dataloader import AuroraBPECG_reg_Dataset
from clef.lightning.lightning_base import BaseModelRegressor


class AuroraBPDataModule(L.LightningDataModule):
    def __init__(
        self,
        ecg_dir,
        batch_size=32,
        num_workers=4,
        input_len=2500,
        task="sbp",  # "sbp" or "dbp"
        task_type="classification",
        protocol="auscultatory",
        use_weighted_sampler=True,
    ):
        super().__init__()
        self.data_path = ecg_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_len = input_len
        self.protocol = protocol
        self.use_weighted_sampler = use_weighted_sampler
        self.lead_num = 1
        self.task = task
        self.task_type = task_type

    def prepare_data(self):
        """Load participant metadata"""
        participants_file = os.path.join(self.data_path, 'participants.tsv')
        if not os.path.exists(participants_file):
            raise FileNotFoundError(f"Participants file not found: {participants_file}")
        
        feature_file = os.path.join(self.data_path, 'features.tsv')
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Features file not found: {feature_file}")

        ppt_df = pd.read_csv(participants_file, delimiter='\t')
        feat_df = pd.read_csv(feature_file, delimiter='\t')
        comb_df = ppt_df.merge(feat_df, how='left', left_on='pid', right_on='pid')

        self.participants_df = comb_df
        print(f"Loaded {len(self.participants_df)} participants from metadata")

        valid_participants = self.participants_df.dropna(subset=['baseline_sbp', 'baseline_dbp'])
        unique_pids = valid_participants['pid'].unique()
        
        # Use the same random state as in the original dataset classes
        self.train_pids, temp_pids = train_test_split(unique_pids, test_size=0.3, random_state=42)
        self.val_pids, self.test_pids = train_test_split(temp_pids, test_size=0.5, random_state=42)

    def _get_split_data(self, pids):
        """Get participant data for specific PIDs"""
        valid_participants = self.participants_df.dropna(subset=['baseline_sbp', 'baseline_dbp'])
        return valid_participants[valid_participants['pid'].isin(pids)]

    def setup(self, stage=None):
        """Setup train/val/test datasets"""
        if not hasattr(self, 'participants_df'):
            self.prepare_data()

        train_participants = self._get_split_data(self.train_pids)

        if self.task_type == "regression":
            self.scaler = StandardScaler()
            
            # Fit scaler on training data targets
            if self.task == "sbp":
                train_participants["baseline_sbp"] = self.scaler.fit_transform(train_participants[["baseline_sbp"]])
                self.participants_df["baseline_sbp"] = self.scaler.transform(self.participants_df[["baseline_sbp"]])
            else:  # dbp
                train_participants["baseline_dbp"] = self.scaler.fit_transform(train_participants[["baseline_dbp"]])
                self.participants_df["baseline_dbp"] = self.scaler.transform(self.participants_df[["baseline_dbp"]])

        if self.task_type == "classification":
            raise ValueError("AuroraBP dataset only support regression tasks.")
        else:
            DatasetClass = AuroraBPECG_reg_Dataset
        
        # Create datasets for each split
        if stage == "fit" or stage is None:
            self.train_dataset = DatasetClass(
                data_path=self.data_path,
                participants_df=self.participants_df,
                stage="train",
                input_len=self.input_len,
                protocol=self.protocol,
                task=self.task
            )

            self.val_dataset = DatasetClass(
                data_path=self.data_path,
                participants_df=self.participants_df,
                stage="val",
                input_len=self.input_len,
                protocol=self.protocol,
                task=self.task
            )
            
            if self.task_type == "classification":
                if self.use_weighted_sampler:
                    self.train_sampler = self._create_weighted_sampler(self.train_dataset)
            else:
                self.train_sampler = None

        if stage == "test" or stage is None:
            self.test_dataset = DatasetClass(
                data_path=self.data_path,
                participants_df=self.participants_df,
                stage="test",
                input_len=self.input_len,
                protocol=self.protocol,
                task=self.task
            )

    def _create_weighted_sampler(self, dataset):
        """Create a weighted sampler to handle class imbalance"""
        labels = [sample['hypertensive'] for sample in dataset.samples]
        
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        print(f"Training set class distribution: {class_counts}")
        print(f"Normal: {class_counts.get(0, 0)}, Hypertensive: {class_counts.get(1, 0)}")
        
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

    def inverse_transform_targets(self, targets):
        """Inverse transform targets back to original scale"""
        if self.scaler is not None and self.task_type == "regression":
            if isinstance(targets, torch.Tensor):
                targets_np = targets.detach().cpu().numpy()
            else:
                targets_np = targets
            
            return self.scaler.inverse_transform(targets_np.reshape(-1, 1)).flatten()
        return targets



class AuroraBPRegressor(BaseModelRegressor):
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
        self.output_dim = num_classes
        self.use_autocast = use_autocast
        self.exp_name = f"aurorabp_{datamodule.task}_{backbone.name}_{datamodule.lead_num}lead"
        
        if do_proj:
            self.projection = nn.Linear(feature_dim, num_classes)
        else:
            self.projection = nn.Identity()

        self.criterion = nn.L1Loss()

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
        """Forward pass implementation following the same pattern as AuroraBPClassifier"""
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