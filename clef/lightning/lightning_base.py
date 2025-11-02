import os
import lightning as L
import torch
import torch.optim as optim
import torchmetrics
from torchmetrics.classification import (
    BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAUROC,
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassAUROC,
    MultilabelAccuracy, MultilabelF1Score, MultilabelPrecision, MultilabelRecall, MultilabelAUROC
)

from typing import Any
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    recall_score, 
    precision_score, 
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    hamming_loss, 
    jaccard_score
)

logger = logging.getLogger(__name__)


class BaseModelClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.criterion = None 

        self.is_binary = self._is_binary_classification()
        self.is_multitask = self._is_multitask_classification()

        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None

    def setup(self, stage=None):
        """Initialize metrics based on classification type"""
        if self.num_classes is None:
            return  
        
        if self.is_multitask:
            # Multi-label metrics (multiple binary tasks)
            self.train_metrics = torchmetrics.MetricCollection({
                'acc': MultilabelAccuracy(num_labels=self.num_classes),
                'f1': MultilabelF1Score(num_labels=self.num_classes),
                'precision': MultilabelPrecision(num_labels=self.num_classes),
                'recall': MultilabelRecall(num_labels=self.num_classes),
                'auroc': MultilabelAUROC(num_labels=self.num_classes)
            })
            self.val_metrics = self.train_metrics.clone(prefix='val_')
            self.test_metrics = self.train_metrics.clone(prefix='test_')
        elif self.is_binary:
            # Binary classification metrics
            self.train_metrics = torchmetrics.MetricCollection({
                'acc': BinaryAccuracy(),
                'f1': BinaryF1Score(),
                'precision': BinaryPrecision(),
                'recall': BinaryRecall(),
                'auroc': BinaryAUROC()
            })
            self.val_metrics = self.train_metrics.clone(prefix='val_')
            self.test_metrics = self.train_metrics.clone(prefix='test_')
        else:
            # Multiclass classification metrics
            self.train_metrics = torchmetrics.MetricCollection({
                'acc': MulticlassAccuracy(num_classes=self.num_classes),
                'f1': MulticlassF1Score(num_classes=self.num_classes),
                'precision': MulticlassPrecision(num_classes=self.num_classes),
                'recall': MulticlassRecall(num_classes=self.num_classes),
                'auroc': MulticlassAUROC(num_classes=self.num_classes)
            })
            self.val_metrics = self.train_metrics.clone(prefix='val_')
            self.test_metrics = self.train_metrics.clone(prefix='test_')

    def capture_moirai_encoder_output(self, x):
        """Capture encoder output using forward hook"""
        
        encoder_output = None
        
        def hook_fn(module, input, output):
            nonlocal encoder_output
            encoder_output = output

        hook = self.model.module.encoder.register_forward_hook(hook_fn)
        
        try:
            inputs = self.model.describe_inputs(x.shape[0]).zeros()
            inputs['past_target'] = x.transpose(1,2)
            
            for key in inputs.keys():
                inputs[key] = inputs[key].to(x.device)
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            return encoder_output
        
        finally:
            hook.remove()

    def forward(self, x):
        """Each subclass should implement its own forward method"""
        raise NotImplementedError("Subclasses must implement forward method")

    def configure_optimizers(self):
        """
        Configure optimizers and schedulers.
        Returns optimizer or a tuple of (optimizers, schedulers).
        """
        # Set default values if they don't exist
        if not hasattr(self.hparams, "optimizer"):
            self.hparams.optimizer = "adam"
        if not hasattr(self.hparams, "lr"):
            self.hparams.lr = 1e-4
        if not hasattr(self.hparams, "weight_decay"):
            self.hparams.weight_decay = 1e-5

        # Select optimizer
        if self.hparams.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:  # default to SGD
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )

        # Return if no scheduler is specified
        if not hasattr(self.hparams, "scheduler") or self.hparams.scheduler is None:
            return optimizer

        # Configure scheduler
        if self.hparams.scheduler.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
        elif self.hparams.scheduler.lower() == "cosinewarmup":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.trainer.max_epochs // 2, T_mult=1, eta_min=0.0
            )
        elif self.hparams.scheduler.lower() == "linear":
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.trainer.max_epochs,
            )
        elif self.hparams.scheduler.lower() == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif self.hparams.scheduler.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif self.hparams.scheduler.lower() == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, verbose=True, min_lr=1e-6
            )
        elif self.hparams.scheduler.lower() == "onecycle":
            steps_per_epoch = (
                self.trainer.estimated_stepping_batches // self.trainer.max_epochs
            )
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr * 10,
                epochs=self.trainer.max_epochs,
                total_steps=self.trainer.estimated_stepping_batches,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                div_factor=25.0,
                three_phase=False,
            )
        else:
            return optimizer

        scheduler_config = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

        # Special case for ReduceLROnPlateau
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_config["monitor"] = "val_loss"

        return [optimizer], [scheduler_config]

    def training_step(self, batch, batch_idx):
        """
        Training step with loss and metrics logging.
        """
        x, y = batch
        
        if isinstance(y, dict):
            y = y['label']
        
        if self.is_binary:
            y = y.float()  # Keep as float for binary classification
        else:
            y = y.long()  # Convert to long for multiclass classification

        use_autocast = hasattr(self, "use_autocast") and self.use_autocast and torch.cuda.is_available()
        
        if use_autocast:
            with torch.autocast(device_type="cuda"):
                logits = self(x)
                loss = self.criterion(logits, y)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)

        if self.train_metrics is not None:
            if self.is_multitask:
                # For multi-label, we need to convert logits to probabilities
                probs = torch.sigmoid(logits)
                metrics = self.train_metrics(probs, y.long())
            elif self.is_binary:
                # For binary, pass logits and targets
                metrics = self.train_metrics(logits, y)
            else:
                # For multiclass, pass logits and class indices
                metrics = self.train_metrics(logits, y)
            
            for name, value in metrics.items():
                self.log(name, value, prog_bar=True, on_step=False, on_epoch=True)
        
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )

        if self.trainer.global_step > 0:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("learning_rate", lr, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Common validation step across all models."""
        x, y = batch
        
        if isinstance(y, dict):
            y = y['label']
        
        if self.is_binary:
            y = y.float()
        else:
            y = y.long()

        use_autocast = hasattr(self, "use_autocast") and self.use_autocast and torch.cuda.is_available()
        
        if use_autocast:
            with torch.autocast(device_type="cuda"):
                logits = self(x)
                loss = self.criterion(logits, y)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)
        
        if self.val_metrics is not None:
            if self.is_multitask:
                # For multi-label, convert logits to probabilities
                probs = torch.sigmoid(logits)
                self.val_metrics(probs, y.long())
            # elif self.is_binary:
            #     self.val_metrics(logits, y)
            else:
                self.val_metrics(logits, y)

        # Log loss
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        output = {
            "val_loss": loss.detach().cpu(),
            "val_logits": logits.detach().cpu(),
            "val_targets": y.detach().cpu(),
        }
        
        self.validation_step_outputs.append(output)
        
        return output

    def test_step(self, batch, batch_idx):
        """Common test step across all models."""
        x, y = batch
        
        if isinstance(y, dict):
            y = y['label']
        
        if self.is_binary:
            y = y.float()
        else:
            y = y.long()

        use_autocast = hasattr(self, "use_autocast") and self.use_autocast and torch.cuda.is_available()
        
        if use_autocast:
            with torch.autocast(device_type="cuda"):
                logits = self(x)
                loss = self.criterion(logits, y)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)

        if self.test_metrics is not None:
            if self.is_multitask:
                # For multi-label, convert logits to probabilities
                probs = torch.sigmoid(logits)
                self.test_metrics(probs, y.long())
            # elif self.is_binary:
            #     self.test_metrics(logits, y)
            else:
                self.test_metrics(logits, y)
        
        # Log loss
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

        output = {
            "test_logits": logits.detach().cpu(),
            "test_targets": y.detach().cpu(),
        }
        
        self.test_step_outputs.append(output)
        
        return output

    def on_validation_epoch_end(self):
        """Process validation results after each epoch."""
        # Log all validation metrics
        if self.val_metrics is not None:
            metrics = self.val_metrics.compute()
            for name, value in metrics.items():
                self.log(name, value, prog_bar=True)
            self.val_metrics.reset()
        
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        """Process test results after the test epoch."""
        if self.test_metrics is not None:
            metrics = self.test_metrics.compute()
            for name, value in metrics.items():
                self.log(name, value, prog_bar=True)
                
            logger.info("\nTest Metrics:")
            for name, value in metrics.items():
                logger.info(f"{name}: {value.item():.4f}")
            
            self.test_metrics.reset()
        
        if len(self.test_step_outputs) > 0:
            # Aggregate all outputs for detailed metrics calculation
            all_logits = torch.cat([x["test_logits"] for x in self.test_step_outputs])
            all_targets = torch.cat([x["test_targets"] for x in self.test_step_outputs])
            
            if self.is_multitask:
                metrics_dict = self._log_detailed_multilabel_metrics(all_logits, all_targets)
            elif self.is_binary:
                metrics_dict = self._log_detailed_binary_metrics(all_logits, all_targets)
            else:
                metrics_dict = self._log_detailed_multiclass_metrics(all_logits, all_targets)
        
            self._save_results_to_csv(metrics_dict)
    
        self.test_step_outputs.clear()

    def _save_results_to_csv(self, metrics_dict):
        """Save results to CSV file"""
        output_file = "results.csv"
        
        exp_name = getattr(self, "exp_name", f"exp_{self.__class__.__name__}")
        
        results_dict = {
            "row_name": exp_name,
        }
        
        results_dict.update(metrics_dict)
        
        results_df = pd.DataFrame([results_dict])
        file_exists = os.path.exists(output_file)
        
        results_df.to_csv(output_file, mode='a', header=not file_exists, index=False)
        
        logger.info(f"Results {'appended to' if file_exists else 'written to'} {output_file}")

    def _log_detailed_multilabel_metrics(self, logits, targets):
        """Log detailed metrics for multilabel classification and return metrics dict"""
        np_probs = torch.sigmoid(logits.float()).numpy()
        np_preds = (np_probs > 0.5).astype(int)
        np_targets = targets.numpy().astype(int)
        
        h_loss = hamming_loss(np_targets, np_preds)
        j_score = jaccard_score(np_targets, np_preds, average='macro', zero_division=0)
        
        acc = accuracy_score(np_targets.flatten(), np_preds.flatten())
        f1_macro = f1_score(np_targets, np_preds, average='macro', zero_division=0)
        f1_micro = f1_score(np_targets, np_preds, average='micro', zero_division=0)
        precision = precision_score(np_targets, np_preds, average='macro', zero_division=0)
        recall = recall_score(np_targets, np_preds, average='macro', zero_division=0)
        try:
            auroc = roc_auc_score(np_targets, np_probs, average='macro')
        except Exception:
            auroc = np.nan
        
        logger.info("\nMultilabel Classification Results:")
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1 Score (Macro): {f1_macro:.4f}")
        logger.info(f"F1 Score (Micro): {f1_micro:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"AUROC: {auroc:.4f}")
        logger.info(f"Hamming Loss: {h_loss:.4f}")
        logger.info(f"Jaccard Score: {j_score:.4f}")

        metrics_perclass_dict = {}
        exp_name = getattr(self, "exp_name", f"exp_{self.__class__.__name__}")
        metrics_perclass_dict["row_name"] = exp_name
        
        # Per-class metrics
        for i in range(self.num_classes):
            class_targets = np_targets[:, i]
            class_preds = np_preds[:, i]
            class_probs = np_probs[:, i]
            
            class_acc = accuracy_score(class_targets, class_preds)
            class_f1 = f1_score(class_targets, class_preds, zero_division=0)
            class_prec = precision_score(class_targets, class_preds, zero_division=0)
            class_rec = recall_score(class_targets, class_preds, zero_division=0)

            # Only calculate AUROC if both classes are present
            if len(np.unique(class_targets)) > 1:
                class_auroc = roc_auc_score(class_targets, class_probs)
                logger.info(f"Class {i}: Acc={class_acc:.4f}, F1={class_f1:.4f}, Prec={class_prec:.4f}, Rec={class_rec:.4f}, AUROC={class_auroc:.4f}")
                metrics_perclass_dict[f"test_class_{i}_auroc"] = float(class_auroc)
            else:
                logger.info(f"Class {i}: Acc={class_acc:.4f}, F1={class_f1:.4f}, Prec={class_prec:.4f}, Rec={class_rec:.4f}, AUROC=N/A")
        
        do_save = 0
        if do_save:
            perclass_output_file = "perclass_results.csv"
            perclass_df = pd.DataFrame([metrics_perclass_dict])
            file_exists = os.path.exists(perclass_output_file)
            perclass_df.to_csv(perclass_output_file, mode='a', header=not file_exists, index=False)
            logger.info(f"Per-class results {'appended to' if file_exists else 'written to'} {perclass_output_file}")

        return {
            "test_auroc": float(auroc),
            "test_epoch_acc": float(acc),
            "test_f1_macro": float(f1_macro),
            "test_f1_micro": float(f1_micro),
            "test_recall": float(recall),
            "test_precision": float(precision),
            "test_hamming_loss": float(h_loss),
            "test_jaccard": float(j_score)
        }

        # return metrics_dict

    def _log_detailed_binary_metrics(self, logits, targets):
        """Log detailed metrics for binary classification and return metrics dict"""
        np_probs = torch.sigmoid(logits.float()).numpy().flatten()
        np_preds = (np_probs > 0.5).astype(int)
        np_targets = targets.numpy().flatten().astype(int)
        
        acc = accuracy_score(np_targets, np_preds)
        f1_macro = f1_score(np_targets, np_preds, average='macro')
        f1_micro = f1_score(np_targets, np_preds, average='micro')
        precision = precision_score(np_targets, np_preds)
        recall = recall_score(np_targets, np_preds)
        
        # Calculate AUROC if possible
        try:
            auroc = roc_auc_score(np_targets, np_probs)
        except Exception:
            auroc = np.nan
        
        # Calculate confusion matrix
        cm = confusion_matrix(np_targets, np_preds)

        # sensitivity = recall  # For binary, sensitivity is the same as recall
        # specificity = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else np.nan
        
        # Log results
        logger.info("\nBinary Classification Results:")
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1 Score: {f1_macro:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"AUROC: {auroc:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(np_targets, np_preds))
        
        return {
            "test_auroc": float(auroc),
            "test_epoch_acc": float(acc),
            "test_f1_macro": float(f1_macro),
            "test_f1_micro": float(f1_micro),
            "test_recall": float(recall),
            "test_precision": float(precision),
            "test_hamming_loss": 0,  # Not applicable for binary
            "test_jaccard": 0  # Not applicable for binary in this form
        }

    def _log_detailed_multiclass_metrics(self, logits, targets):
        """Log detailed metrics for multiclass classification and return metrics dict"""
        np_preds = torch.argmax(logits, dim=1).numpy()
        np_targets = targets.numpy().astype(int)
        
        acc = accuracy_score(np_targets, np_preds)
        f1_macro = f1_score(np_targets, np_preds, average='macro')
        f1_micro = f1_score(np_targets, np_preds, average='micro')
        precision = precision_score(np_targets, np_preds, average='macro')
        recall = recall_score(np_targets, np_preds, average='macro')
        
        # Calculate one-hot encoded versions for AUROC
        from sklearn.preprocessing import OneHotEncoder
        try:
            encoder = OneHotEncoder(sparse=False)
        except TypeError:
            encoder = OneHotEncoder(sparse_output=False)
        
        np_targets_one_hot = encoder.fit_transform(np_targets.reshape(-1, 1))
        
        # For multiclass, we need class probabilities
        np_probs = torch.softmax(logits.float(), dim=1).numpy()
        
        try:
            auroc = roc_auc_score(np_targets_one_hot, np_probs, multi_class='ovr')
        except Exception:
            auroc = np.nan
        
        cm = confusion_matrix(np_targets, np_preds)
        
        # Log results
        logger.info("\nMulticlass Classification Results:")
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1 Score (Macro): {f1_macro:.4f}")
        logger.info(f"F1 Score (Micro): {f1_micro:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"AUROC: {auroc:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(np_targets, np_preds))
        
        return {
            "test_auroc": float(auroc),
            "test_epoch_acc": float(acc),
            "test_f1_macro": float(f1_macro),
            "test_f1_micro": float(f1_micro),
            "test_recall": float(recall),
            "test_precision": float(precision),
            "test_hamming_loss": 0,  # Not applicable for multiclass
            "test_jaccard": 0  # Not applicable for multiclass in this form
        }

    def _is_binary_classification(self) -> bool:
        if hasattr(self, "criterion"):
            return isinstance(self.criterion, (torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss))
            
        return False
    
    def _is_multitask_classification(self) -> bool:
        return False

    def _do_perclass_accuracy(self) -> bool:
        return False

    def log_with_prog_bar(self, name: str, value: Any, prog_bar: bool = True) -> None:
        """Utility method for logging with prog_bar=True by default"""
        self.log(name, value, prog_bar=prog_bar)



class BaseModelRegressor(L.LightningModule):
    def __init__(self, datamodule=None):
        super().__init__()
        # Initialize tracking collections for outputs
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.criterion = None  # Will be set by child classes
        self.datamodule = datamodule

    def capture_moirai_encoder_output(self, x):
        """Capture encoder output using forward hook"""
        
        encoder_output = None
        
        def hook_fn(module, input, output):
            nonlocal encoder_output
            encoder_output = output

        hook = self.model.module.encoder.register_forward_hook(hook_fn)
        
        try:
            inputs = self.model.describe_inputs(x.shape[0]).zeros()
            inputs['past_target'] = x.transpose(1,2)
            
            for key in inputs.keys():
                inputs[key] = inputs[key].to(x.device)
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            return encoder_output
        
        finally:
            hook.remove()

    def forward(self, x):
        """Each subclass should implement its own forward method"""
        raise NotImplementedError("Subclasses must implement forward method")

    def configure_optimizers(self):
        """
        Configure optimizers and schedulers.
        Returns optimizer or a tuple of (optimizers, schedulers).
        """
        # Set default values if they don't exist
        if not hasattr(self.hparams, "optimizer"):
            self.hparams.optimizer = "adam"
        if not hasattr(self.hparams, "lr"):
            self.hparams.lr = 1e-4
        if not hasattr(self.hparams, "weight_decay"):
            self.hparams.weight_decay = 1e-5

        # Select optimizer
        if self.hparams.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:  # default to SGD
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )

        if not hasattr(self.hparams, "scheduler") or self.hparams.scheduler is None:
            return optimizer

        # Configure scheduler
        if self.hparams.scheduler.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
        elif self.hparams.scheduler.lower() == "cosinewarmup":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.trainer.max_epochs // 2, T_mult=1, eta_min=0.0
            )
        elif self.hparams.scheduler.lower() == "linear":
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.trainer.max_epochs,
            )
        elif self.hparams.scheduler.lower() == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif self.hparams.scheduler.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif self.hparams.scheduler.lower() == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, verbose=True
            )
        elif self.hparams.scheduler.lower() == "onecycle":
            steps_per_epoch = (
                self.trainer.estimated_stepping_batches // self.trainer.max_epochs
            )
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr * 10,
                epochs=self.trainer.max_epochs,
                total_steps=self.trainer.estimated_stepping_batches,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                div_factor=25.0,
                three_phase=False,
            )
        else:
            return optimizer

        scheduler_config = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

        # Special case for ReduceLROnPlateau
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_config["monitor"] = "val_loss"

        return [optimizer], [scheduler_config]

    def training_step(self, batch, batch_idx):
        """
        Training step with loss and metrics logging.
        """
        x, y = batch
        
        if isinstance(y, dict):
            # Some datasets might have other metadata in the batch
            y = y['label']
            
        y = y.float()  # Ensure target is float for regression

        use_autocast = hasattr(self, "use_autocast") and self.use_autocast and torch.cuda.is_available()
        
        if use_autocast:
            with torch.autocast(device_type="cuda"):
                pred = self(x)
                loss = self.criterion(pred.squeeze(), y.squeeze())
        else:
            pred = self(x)
            loss = self.criterion(pred.squeeze(), y.squeeze())

        mae = torch.mean(torch.abs(pred.squeeze() - y.squeeze()))
        mse = torch.mean(torch.pow(pred.squeeze() - y.squeeze(), 2))
        
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_mae",
            mae,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_mse",
            mse,
            prog_bar=False,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

        if self.trainer.global_step > 0:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("learning_rate", lr, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Common validation step across all regression models."""
        x, y = batch
        
        if isinstance(y, dict):
            y = y['label']
            
        y = y.float()  # Ensure target is float

        # Apply autocast if available
        use_autocast = hasattr(self, "use_autocast") and self.use_autocast and torch.cuda.is_available()
        
        if use_autocast:
            with torch.autocast(device_type="cuda"):
                pred = self(x)
                loss = self.criterion(pred.squeeze(), y.squeeze())
        else:
            pred = self(x)
            loss = self.criterion(pred.squeeze(), y.squeeze())

        mae = torch.mean(torch.abs(pred.squeeze() - y.squeeze()))
        mse = torch.mean(torch.pow(pred.squeeze() - y.squeeze(), 2))

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)
        self.log("val_mse", mse, prog_bar=False)

        output = {
            "val_loss": loss.detach().cpu(),
            "val_pred": pred.detach().cpu(),
            "val_target": y.detach().cpu(),
            "val_mae": mae.detach().cpu(),
        }
        if output["val_pred"].isnan().any():
            logger.warning("NaN values found in validation predictions.")
        self.validation_step_outputs.append(output)
        
        return output

    def test_step(self, batch, batch_idx):
        """Common test step across all regression models."""
        x, y = batch
        
        if isinstance(y, dict):
            y = y['label']
            
        y = y.float()  # Ensure target is float

        # Apply autocast if available
        use_autocast = hasattr(self, "use_autocast") and self.use_autocast and torch.cuda.is_available()
        
        if use_autocast:
            with torch.autocast(device_type="cuda"):
                pred = self(x)
        else:
            pred = self(x)

        mae = torch.mean(torch.abs(pred.squeeze() - y.squeeze()))
        mse = torch.mean(torch.pow(pred.squeeze() - y.squeeze(), 2))

        self.log("test_mae", mae, prog_bar=True)
        self.log("test_mse", mse, prog_bar=True)

        output = {
            "test_pred": pred.detach().cpu(),
            "test_target": y.detach().cpu(),
            "test_mae": mae.detach().cpu(),
            "test_mse": mse.detach().cpu(),
        }
        
        self.test_step_outputs.append(output)
        
        return output

    def on_validation_epoch_end(self):
        """Process validation results after each epoch."""
        if not hasattr(self, "validation_step_outputs") or len(self.validation_step_outputs) == 0:
            logger.warning("No validation outputs found. Skipping validation metrics calculation.")
            return

        all_preds = torch.cat([x["val_pred"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["val_target"] for x in self.validation_step_outputs])
        all_losses = torch.stack([x["val_loss"] for x in self.validation_step_outputs])
        
        np_preds = all_preds.float().numpy()
        np_targets = all_targets.numpy()
        
        mae = mean_absolute_error(np_targets, np_preds)
        mse = mean_squared_error(np_targets, np_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(np_targets, np_preds)
        
        # Log all metrics
        avg_loss = torch.mean(all_losses).item()
        self.log("val_epoch_loss", avg_loss, prog_bar=True)
        self.log("val_epoch_mae", mae, prog_bar=True)
        self.log("val_rmse", rmse, prog_bar=True)
        self.log("val_r2", r2, prog_bar=False)
        
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        """Process test results after the test epoch."""
        if not hasattr(self, "test_step_outputs") or len(self.test_step_outputs) == 0:
            logger.warning("No test outputs found. Skipping test metrics calculation.")
            return
            
        all_preds = torch.cat([x["test_pred"] for x in self.test_step_outputs])
        all_targets = torch.cat([x["test_target"] for x in self.test_step_outputs])
        
        np_preds = all_preds.float().numpy().flatten()
        np_targets = all_targets.numpy().flatten()

        if self.datamodule and hasattr(self.datamodule, 'inverse_transform_targets'):
            # Transform predictions and targets back to original scale
            np_preds = self.datamodule.inverse_transform_targets(np_preds)
            np_targets = self.datamodule.inverse_transform_targets(np_targets)
        
        mae = mean_absolute_error(np_targets, np_preds)
        mse = mean_squared_error(np_targets, np_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(np_targets, np_preds)
        
        # Log all metrics
        self.log("test_epoch_mae", mae, prog_bar=True)
        self.log("test_rmse", rmse, prog_bar=True)
        self.log("test_r2", r2, prog_bar=True)
        self.log("test_mse", mse, prog_bar=False)
        
        # Print summary
        logger.info("\nRegression Model Performance:")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"R²: {r2:.4f}")
        
        # Calculate error distribution statistics
        errors = np_preds - np_targets
        logger.info("\nError Distribution:")
        logger.info(f"Mean Error: {np.mean(errors):.4f}")
        logger.info(f"Error Std: {np.std(errors):.4f}")
        logger.info(f"Median Error: {np.median(errors):.4f}")
        logger.info(f"Min Error: {np.min(errors):.4f}")
        logger.info(f"Max Error: {np.max(errors):.4f}")

        metrics_dict = {
            "test_mae": float(mae),
            "test_rmse": float(rmse),
            "test_r2": float(r2),
            "test_mse": float(mse),
            "test_mean_error": float(np.mean(errors)),
            "test_error_std": float(np.std(errors)),
            "test_median_error": float(np.median(errors)),
            "test_min_error": float(np.min(errors)),
            "test_max_error": float(np.max(errors))
        }
        
        self._save_results_to_csv(metrics_dict)
        
        self.test_step_outputs.clear()
    
    def _save_results_to_csv(self, metrics_dict):
        """Save results to CSV file"""
        output_file = "reg_results.csv"
        
        exp_name = getattr(self, "exp_name", f"exp_{self.__class__.__name__}")
        
        results_dict = {
            "row_name": exp_name,
        }
        
        results_dict.update(metrics_dict)
        results_df = pd.DataFrame([results_dict])
        file_exists = os.path.exists(output_file)
        
        results_df.to_csv(output_file, mode='a', header=not file_exists, index=False)
        
        logger.info(f"Results {'appended to' if file_exists else 'written to'} {output_file}")
    
    def log_with_prog_bar(self, name: str, value: Any, prog_bar: bool = True) -> None:
        """Utility method for logging with prog_bar=True by default"""
        self.log(name, value, prog_bar=prog_bar)
