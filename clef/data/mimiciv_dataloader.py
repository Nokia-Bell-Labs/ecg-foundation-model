# MIMIC-IV ECG
import os
import wfdb
import numpy as np
import pandas as pd
from typing import List, Optional, Any, Dict
import neurokit2 as nk

import lightning as L
from torch.utils.data import DataLoader

from clef.data.contrastive_dataloader import BaseECGContrastiveDataset
from clef.utils.utils import timing


class MIMICIVECGContrastiveDataset(BaseECGContrastiveDataset):
    def __init__(
        self,
        data_path: str,
        metadata_path: Optional[str] = None,
        leads: Optional[List[str]] = None,
        input_len: int = 2500,
        normalization: bool = True,
        transform: Optional = None,
        simclr: bool = True,
        use_metadata: bool = True,
        metadata_attributes: Optional[List[str]] = None,
    ):
        self.input_leads = [
            "I",
            "II",
            "III",
            "aVR",
            "aVF",
            "aVL",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]

        # If leads are provided as strings, convert to indices
        if leads is not None and isinstance(leads[0], str):
            self.leads = [self.input_leads.index(lead) for lead in leads]
        else:
            self.leads = leads

        self.use_metadata = use_metadata
        self.metadata_attributes = metadata_attributes if metadata_attributes else [
            'gender', 'anchor_age', 'BMI (kg/m2)', 'Blood Pressure', 'Height (Inches)', 'Weight (Lbs)', 'age_at_ecg'
        ]

        self.metadata_df = None # initialise
        if metadata_path and use_metadata:
            self.metadata_df = self._read_metadata(metadata_path)

        super().__init__(
            data_path=data_path,
            metadata_path=metadata_path,
            leads=self.leads,
            input_len=input_len,
            normalization=normalization,
            transform=transform,
            simclr=simclr,
            use_metadata=use_metadata,
            metadata_attributes=self.metadata_attributes,
        )
    
    def _find_files(self) -> List[str]:
        """Find all available record IDs in the dataset"""
        # file_csv = self.data_path + "record_list.csv"
        file_csv = "dataset/mimic-iv/patientrecord_list.csv"
        file_list = pd.read_csv(file_csv)
        
        # files = file_list.head(384)["path"].tolist()
        files = file_list["path"].tolist()
        return files

    def _read_header(self, hash_file_name: str) -> Any:
        """Read the header for a given record"""
        file_path = os.path.join(self.data_path, hash_file_name)
        return wfdb.rdheader(file_path)

    @timing(enabled=False)
    def _read_partial_record(self, fname: str, start: int, end: int) -> np.ndarray:
        file_path = os.path.join(self.data_path, fname)
        record = wfdb.rdrecord(
            file_path,
            channels=self.leads if self.leads is not None else None,
            sampfrom=start,
            sampto=end,
        )
        return record.p_signal
    
    def _read_metadata(self, fname: str) -> pd.DataFrame:
        meta_df = pd.read_csv(fname)
        # set path as index
        if 'path' in meta_df.columns:
            meta_df.set_index('path', inplace=True)
        return meta_df

    def _get_signal_length(self, header: Any) -> int:
        return header.sig_len

    def _get_metadata_for_record(self, fname: str) -> np.ndarray:
        """Get metadata for a specific record as a numpy array"""
        # gender, age, BMI, systolic, diastolic, height, weight
        if not self.use_metadata or self.metadata_df is None:
            return np.zeros(self._get_metadata_feature_dim())
        
        try:
            metadata = self.metadata_df.loc[fname]
            features = []
            
            # Process each attribute in a consistent order
            # Gender (binary: 0=female, 1=male)
            if 'gender' in metadata and not pd.isna(metadata['gender']):
                features.append(0.0 if metadata['gender'] == 'F' else 1.0)
            else:
                features.append(np.nan)
                
            # Age
            if 'age_at_ecg' in metadata and not pd.isna(metadata['age_at_ecg']):
                features.append(float(metadata['age_at_ecg']))
            elif 'anchor_age' in metadata and not pd.isna(metadata['anchor_age']):
                features.append(float(metadata['anchor_age']))
            else:
                features.append(np.nan)
                
            # BMI
            if 'BMI (kg/m2)' in metadata and not pd.isna(metadata['BMI (kg/m2)']):
                features.append(float(metadata['BMI (kg/m2)']))
            else:
                features.append(np.nan)
                
            # Blood pressure (systolic & diastolic)
            if 'Blood Pressure' in metadata and not pd.isna(metadata['Blood Pressure']) and isinstance(metadata['Blood Pressure'], str):
                try:
                    systolic, diastolic = metadata['Blood Pressure'].split('/')
                    features.append(float(systolic))
                    features.append(float(diastolic))
                except (ValueError, AttributeError):
                    features.append(np.nan)  # systolic
                    features.append(np.nan)  # diastolic
            else:
                features.append(np.nan)  # systolic
                features.append(np.nan)  # diastolic
                
            # Height
            if 'Height (Inches)' in metadata and not pd.isna(metadata['Height (Inches)']):
                features.append(float(metadata['Height (Inches)']))
            else:
                features.append(np.nan)
                
            # Weight
            if 'Weight (Lbs)' in metadata and not pd.isna(metadata['Weight (Lbs)']):
                features.append(float(metadata['Weight (Lbs)']))
            else:
                features.append(np.nan)
        
            # Heart rate and heart rate variability
            if 'heart_rate' in metadata and not pd.isna(metadata['heart_rate']):
                features.append(float(metadata['heart_rate']))
            else:
                features.append(np.nan)

            if 'heart_rate_var' in metadata and not pd.isna(metadata['heart_rate_var']):
                features.append(float(metadata['heart_rate_var']))
            else:
                features.append(np.nan)

            # SCORE 2
            if 'rate_meta' in metadata and not pd.isna(metadata['rate_meta']):
                features.append(float(metadata['rate_meta']))
            else:
                features.append(np.nan)

            if 'score2_risk' in metadata and not pd.isna(metadata['score2_risk']):
                features.append(float(metadata['score2_risk']))
            else:
                features.append(np.nan)

            return np.array(features)
            
        except (KeyError, TypeError) as e:
            # Return array of NaNs if record not found
            return np.full(self._get_metadata_feature_dim(), np.nan)
        
    def _get_metadata_feature_dim(self):
        """Return the number of metadata features in the array"""
        # 7 features: gender, age, BMI, systolic BP, diastolic BP, height, weight
        # 4 additional features: heart rate, heart rate variability, mask rate, SCORE2 risk
        return 11
        

    
class MIMICIVDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        metadata_path: Optional[str] = None,
        batch_size: int = 32,
        leads: Optional[List[str]] = None,
        input_len: int = 2500,
        normalization: bool = True,
        transform: str = None,
        simclr: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
        use_metadata: bool = True,
        **params,
    ):
        super().__init__()
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.leads = leads
        self.input_len = input_len
        self.normalization = normalization
        self.transform = transform
        self.simclr = simclr
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.use_metadata = use_metadata
        self.params = params

    def setup(self, stage=None):  # Create datasets
        self.train_dataset = MIMICIVECGContrastiveDataset(
            data_path=self.data_path,
            metadata_path=self.metadata_path,
            leads=self.leads,
            input_len=self.input_len,
            normalization=self.normalization,
            transform=self.transform,
            simclr=self.simclr,
            use_metadata=self.use_metadata,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
