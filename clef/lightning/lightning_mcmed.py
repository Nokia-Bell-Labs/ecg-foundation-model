import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import lightning as L
from collections import Counter
from typing import Optional
from sklearn.preprocessing import StandardScaler

from clef.data.downstream_dataloader import MCMEDECGDataset
from clef.lightning.lightning_base import BaseModelClassifier, BaseModelRegressor


class MCMEDDataModule(L.LightningDataModule):
    def __init__(
        self,
        ecg_dir,
        label_csv,
        batch_size=32,
        num_workers=4,
        input_len=2500,
        target_fs=500,
        normalize=True,
        task_type="classification",
        task="sbp",
        num_classes=2,
    ):
        super().__init__()
        self.data_path = ecg_dir
        self.label_path = label_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_len = input_len
        self.target_fs = target_fs
        self.normalize = normalize
        self.task_type = task_type
        self.task = task
        self.lead_num = 1  # Lead II
        self.n_classes = num_classes

        task_to_column = {
            "sbp": "Triage_SBP",
            "dbp": "Triage_DBP",
            "ed-dispo": "ED_dispo",
            "dc-dispo": "DC_dispo",
            "acuity": "Triage_acuity"
        }
        
        self.label_column = task_to_column.get(task, "Triage_SBP")
        
        self.task_name = self.label_column

    def setup(self, stage=None):
        """Setup train/val/test datasets"""

        variables_df = pd.read_csv(self.label_path)
        # scaling for regression tasks
        if self.task_type == "regression":
            self.scaler = None
            train_split_file = os.path.join(self.data_path, "split_chrono_train.csv")
            train_split_df = pd.read_csv(train_split_file, header=None)
            train_enc_ids = [str(row.iloc[0]) for _, row in train_split_df.iterrows()]
            train_df = variables_df[variables_df['visit_id'].astype(str).isin(train_enc_ids)]

            if self.task in ["sbp", "dbp"]:
                self.scaler = StandardScaler()
                train_df[[self.label_column]] = self.scaler.fit_transform(train_df[[self.label_column]])
                variables_df[[self.label_column]] = self.scaler.transform(variables_df[[self.label_column]])
        
        # Create datasets for each split
        if stage == "fit" or stage is None:
            self.train_dataset = MCMEDECGDataset(
                data_path=self.data_path,
                label_df=variables_df,
                stage="train",
                input_len=self.input_len,
                target_fs=self.target_fs,
                normalize=self.normalize,
                task=self.task_type,
                label_column=self.label_column
            )
            
            self.val_dataset = MCMEDECGDataset(
                data_path=self.data_path,
                label_df=variables_df,
                stage="val",
                input_len=self.input_len,
                target_fs=self.target_fs,
                normalize=self.normalize,
                task=self.task_type,
                label_column=self.label_column
            )

        if stage == "test" or stage is None:
            self.test_dataset = MCMEDECGDataset(
                data_path=self.data_path,
                label_df=variables_df,
                stage="test",
                input_len=self.input_len,
                target_fs=self.target_fs,
                normalize=self.normalize,
                task=self.task_type,
                label_column=self.label_column
            )

    def _create_weighted_sampler(self, dataset):
        """Create a weighted sampler to handle class imbalance"""
        if hasattr(dataset, 'get_class_distribution') and dataset.task == "classification":
            class_dist = dataset.get_class_distribution()
            labels = [sample['label'] for sample in dataset.samples]
            class_counts = class_dist['class_counts']
        else:
            # Fallback if class distribution method not available
            labels = [0] * len(dataset) 
            class_counts = Counter(labels)
        
        total_samples = len(labels)
        
        print(f"Training set class distribution: {class_counts}")
        
        # Calculate weights inversely proportional to class frequency
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
        sample_weights = [class_weights.get(label, 1.0) for label in labels]
        
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

    def inverse_transform_targets(self, targets):
        """Inverse transform targets back to original scale"""
        if self.scaler is not None and self.task_type == "regression":
            if isinstance(targets, torch.Tensor):
                targets_np = targets.detach().cpu().numpy()
            else:
                targets_np = targets
            
            return self.scaler.inverse_transform(targets_np.reshape(-1, 1)).flatten()
        return targets


class MCMEDClassifier(BaseModelClassifier):
    def __init__(
        self,
        backbone,
        datamodule,
        do_proj=False,
        num_classes=2,
        lr=1e-4,
        weight_decay=1e-5,
        optimizer: str = "adam",
        scheduler: Optional[str] = "cosinewarmup",
        feature_dim=768,
        use_weighted_loss=False,
        use_autocast=True,
        task_type="classification",  # "classification" or "regression"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "datamodule"])
        self.model = backbone
        self.num_classes = num_classes
        self.use_autocast = use_autocast
        self.task_type = task_type
        task_name = datamodule.task_name.replace("_", "")
        self.exp_name = f"mcmed_{task_name}_{backbone.name}_{datamodule.lead_num}lead"
        
        if do_proj:
            if task_type == "regression":
                self.projection = nn.Linear(feature_dim, 1)  # Single output for regression
            else:
                self.projection = nn.Linear(feature_dim, num_classes)
        else:
            self.projection = nn.Identity()

        # Setup loss function based on task type and configuration
        if task_type == "regression":
            self.criterion = nn.L1Loss()
            print("Using MSE Loss for regression")
        else:
            if use_weighted_loss and hasattr(datamodule, 'train_dataset'):
                # Calculate class weights from training data if available
                try:
                    if hasattr(datamodule.train_dataset, 'get_class_distribution'):
                        class_dist = datamodule.train_dataset.get_class_distribution()
                        class_counts = Counter([class_dist.get(i, 0) for i in range(num_classes)])
                    else:
                        # Fallback: assume balanced classes
                        class_counts = Counter({i: 1 for i in range(num_classes)})
                    
                    total_samples = sum(class_counts.values())
                    weight_tensor = torch.tensor([
                        total_samples / class_counts.get(i, 1) for i in range(num_classes)
                    ], dtype=torch.float)
                    
                    self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
                    print(f"Using weighted CrossEntropyLoss with weights: {weight_tensor}")
                except Exception as e:
                    print(f"Failed to create weighted loss, using standard CrossEntropyLoss: {e}")
                    self.criterion = nn.CrossEntropyLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
                print("Using standard CrossEntropyLoss")

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
        """Forward pass implementation following the same pattern as other classifiers"""
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
        """Returns True if this is binary classification"""
        return False
    
    def _is_multitask_classification(self) -> bool:
        return False  # Set to True if you have multi-label classification

    def _is_regression(self) -> bool:
        """Returns True if this is a regression task"""
        return self.task_type == "regression"

    def get_class_distribution(self):
        """Get class distribution information for analysis"""
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'datamodule'):
            if hasattr(self.trainer.datamodule, 'train_dataset'):
                if hasattr(self.trainer.datamodule.train_dataset, 'get_class_distribution'):
                    return self.trainer.datamodule.train_dataset.get_class_distribution()
        return None


class MCMEDRegressor(BaseModelRegressor):
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
        # remove the _ from task name
        task_name = datamodule.task_name.replace("_", "")
        self.exp_name = f"mcmed_reg_{task_name}_{backbone.name}_{datamodule.lead_num}lead"

        if do_proj:
            self.projection = nn.Linear(feature_dim, num_classes)
        else:
            self.projection = nn.Identity()

        # Use MSE loss for regression
        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        print("Using MSE Loss for regression")

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
        """Forward pass implementation following the same pattern as MCMEDClassifier"""
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