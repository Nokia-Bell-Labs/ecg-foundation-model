import os
import csv
import glob
import numpy as np
import pandas as pd
import wfdb
from wfdb import processing
import random
import torch
import torch.nn.functional as F
import json
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from clef.utils.utils import is_flat_signal, is_valid_signal
from clef.utils.utils import load_records

def z_score_normalization(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

def z_score(signal):
    new_signal = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        new_signal[i, :] = z_score_normalization(signal[i, :])
    return new_signal

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''
    Usage: bandpass_filter(stacked_ecg_data, 0.67, 40, fs=500)
    '''
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")

    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i, :] = filtfilt(b, a, data[i, :])

    return filtered_data


class AuroraBPECG_reg_Dataset(Dataset):
    def __init__(self, data_path, participants_df, stage="train", input_len=5000, protocol="auscultatory", task="sbp"):
        """
        AuroraBP Dataset for ECG-based blood pressure regression
        
        Args:
            data_path: Path to AuroraBP root directory
            participants_df: DataFrame with participant metadata including sbp/dbp
            stage: 'train', 'val', or 'test'  
            input_len: Length of ECG segment (samples)
        """
        self.data_path = data_path
        self.input_len = input_len
        self.protocol = protocol
        self.task = task  # "sbp" or "dbp"
        
        valid_participants = participants_df.dropna(subset=['baseline_sbp', 'baseline_dbp'])
        
        unique_pids = valid_participants['pid'].unique()
        train_pids, temp_pids = train_test_split(unique_pids, test_size=0.3, random_state=42)
        val_pids, test_pids = train_test_split(temp_pids, test_size=0.5, random_state=42)
        
        if stage == "train":
            selected_pids = train_pids
        elif stage == "val":
            selected_pids = val_pids
        else:
            selected_pids = test_pids
            
        self.participant_data = valid_participants[valid_participants['pid'].isin(selected_pids)]
        
        self.samples = self._find_ecg_files()
        
        print(f"{stage.upper()} set: {len(self.samples)} samples, "
              f"{len(selected_pids)} participants")
        
    def _find_ecg_files(self):
        """Find available ECG TSV files for participants"""
        samples = []
        measurements_dir = os.path.join(self.data_path, f'measurements_{self.protocol}')
        
        for _, row in self.participant_data.iterrows():
            pid = row['pid']
            sbp = row['baseline_sbp']
            dbp = row['baseline_dbp']
            
            tsv_file = os.path.join(measurements_dir, f"{pid}/{pid}.initial.Calibration_start_1.tsv")
            if os.path.exists(tsv_file):
                samples.append({
                    'tsv_file': tsv_file,
                    'pid': pid,
                    'sbp': sbp,
                    'dbp': dbp,
                })
        return samples
    
    def _load_ecg_signal(self, tsv_file):
        """Load ECG signal from TSV file"""
        try:
            signal_df = pd.read_csv(tsv_file, sep='\t')
            ecg_signal = signal_df['ekg'].values
            
            # Remove NaN values
            ecg_signal = ecg_signal[~np.isnan(ecg_signal)]
            
            if len(ecg_signal) == 0:
                return np.zeros(self.input_len)
                
            return ecg_signal
            
        except Exception as e:
            print(f"Error loading {tsv_file}: {e}")
            return np.zeros(self.input_len)
    
    def _process_signal(self, signal):
        """Process ECG signal to fixed length"""
        if len(signal) >= self.input_len:
            # Random crop if signal is longer
            start_idx = np.random.randint(0, len(signal) - self.input_len + 1)
            signal = signal[start_idx:start_idx + self.input_len]
        else:
            # Pad with zeros if signal is shorter
            signal = np.pad(signal, (0, self.input_len - len(signal)), 'constant')
        
        # Z-score normalization
        if np.std(signal) > 0:
            signal = (signal - np.mean(signal)) / np.std(signal)
        
        return signal
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        ecg_signal = self._load_ecg_signal(sample['tsv_file'])
        ecg_signal = self._process_signal(ecg_signal)

        # ecg_signal = bandpass_filter(ecg_signal.transpose(), 0.67, 40, fs=250)
        # signal = z_score_normalization(data)
        
        ecg_tensor = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0)  # [1, input_len]
        
        sbp = torch.tensor(sample['sbp'], dtype=torch.float)
        dbp = torch.tensor(sample['dbp'], dtype=torch.float)

        if self.task == "sbp":
            target = sbp
        else:
            target = dbp
        
        return ecg_tensor, target
                    

class Chapman1LeadDataset(Dataset):
    def __init__(
        self,
        data_path,
        stage="train",
        task="classification",
        input_len=1000,
        transform=None,
        target_fs=100,
        normalize=True,
        lead_idx=1,
    ):
        self.data_path = data_path
        self.stage = stage
        self.task = task
        self.input_len = input_len
        self.transform = transform
        self.target_fs = target_fs
        self.normalize = normalize
        self.lead_idx = lead_idx
        
        # Load preprocessed data
        self.data_file = os.path.join(data_path, f"data/{stage}_data.npy")
        self.labels_file = os.path.join(data_path, f"data/{stage}_labels.npy")
        
        if not os.path.exists(self.data_file) or not os.path.exists(self.labels_file):
            raise FileNotFoundError(
                f"Preprocessed data not found. Please run preprocess/preprocess_chapman.py first.\n"
                f"Expected files: {self.data_file}, {self.labels_file}"
            )
        
        # Load data
        self.ecg_data = np.load(self.data_file)  # Shape: (N, leads, samples)
        self.labels = np.load(self.labels_file, allow_pickle=True)  # Shape: (N, num_classes)
        
        self.class_counts = np.sum(self.labels, axis=0)
        self.class_weights = len(self.labels) / (self.labels.shape[1] * self.class_counts + 1e-8)
        
        # Validate data
        assert len(self.ecg_data) == len(self.labels), \
            f"Data length mismatch: {len(self.ecg_data)} vs {len(self.labels)}"
    
    def __len__(self):
        return len(self.ecg_data)
    
    def __getitem__(self, idx):
        ecg_signal = self.ecg_data[idx, self.lead_idx]  
        label = self.labels[idx]  
        
        signal = np.expand_dims(ecg_signal, axis=0)
        signal = z_score_normalization(signal)
        signal = torch.tensor(signal, dtype=torch.float32)
        
        if self.transform:
            signal = self.transform(signal)
        label = np.array(label, dtype=np.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        return signal, label


class Chapman12LeadDataset(Dataset):
    def __init__(
        self,
        data_path,
        stage="train",
        task="classification",
        input_len=1000,
        transform=None,
        target_fs=100,
        normalize=True,
    ):
        self.data_path = data_path
        self.stage = stage
        self.task = task
        self.input_len = input_len
        self.transform = transform
        self.target_fs = target_fs
        self.normalize = normalize
        
        # Load preprocessed data
        self.data_file = os.path.join(data_path, f"data/{stage}_data.npy")
        self.labels_file = os.path.join(data_path, f"data/{stage}_labels.npy")
        
        if not os.path.exists(self.data_file) or not os.path.exists(self.labels_file):
            raise FileNotFoundError(
                f"Preprocessed data not found. Please run preprocess/preprocess_chapman.py first.\n"
                f"Expected files: {self.data_file}, {self.labels_file}"
            )
        
        # Load data
        self.ecg_data = np.load(self.data_file)  # Shape: (N, leads, samples)
        self.labels = np.load(self.labels_file, allow_pickle=True)  # Shape: (N, num_classes)
        
        self.class_counts = np.sum(self.labels, axis=0)
        self.class_weights = len(self.labels) / (self.labels.shape[1] * self.class_counts + 1e-8)
        
        # Validate data
        assert len(self.ecg_data) == len(self.labels), \
            f"Data length mismatch: {len(self.ecg_data)} vs {len(self.labels)}"
    
    def __len__(self):
        return len(self.ecg_data)
    
    def __getitem__(self, idx):
        ecg_signal = self.ecg_data[idx]  # Shape: (leads, samples)
        label = self.labels[idx]  # Shape: (num_classes,)
        
        signal = z_score(ecg_signal)
        signal = torch.tensor(signal, dtype=torch.float32)
        
        if self.transform:
            signal = self.transform(signal)
        
        label = np.array(label, dtype=np.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        return signal, label
    
    def get_class_weights(self):
        """Return class weights for handling imbalanced data"""
        return torch.tensor(self.class_weights, dtype=torch.float32)
    
    def get_class_counts(self):
        """Return class counts for analysis"""
        return self.class_counts


class IcentiaBeatDataset(Dataset):
    def __init__(self, data_path, patient_ids, stage="train", input_len=1000, transform=None):
        """
        Dataset for ECG beat classification
        
        Args:
            data_path: Root directory containing ECG records
            patient_ids: List of patient IDs to include
            input_len: Length of ECG segment to extract
            transform: Optional transforms to apply
        """
        self.data_path = data_path
        self.patient_ids = patient_ids
        self.stage = stage # test, val, train
        self.input_len = input_len
        self.transform = transform
        
        # Beat labels: N (normal), S (supraventricular), V (ventricular), Q (unclassifiable)
        self.beat_types = ['N', 'S', 'V']
        self.beat_map = {beat: idx for idx, beat in enumerate(self.beat_types)}
        
        # Prepare samples
        self.samples = self._prepare_samples()
        
        # Calculate class weights
        labels = [s['label'] for s in self.samples]
        class_counts = Counter(labels)
        total_samples = len(labels)
        self.class_weight = {cls: total_samples / count for cls, count in class_counts.items()}
        
    def _prepare_samples(self):
        if os.path.exists(f"dataset/Icentia11K/beat/icentia_beat_{self.stage}.json"):
            # If the dataset is already prepared, load it
            with open(f"dataset/Icentia11K/beat/icentia_beat_{self.stage}.json", 'r') as f:
                samples = json.load(f)
            print(f"Loaded {len(samples)} samples from preprocessed dataset.")
            
        else:
            samples = []
            
            # Track which beat types have been found for each patient
            found_beat_types_per_patient = {patient: set() for patient in self.patient_ids}
            
            for patient in self.patient_ids:
                patient_record = f'{self.data_path}/{patient}'[:-1]
                
                segments_file = os.path.join(patient_record, "RECORDS")
                if not os.path.exists(segments_file):
                    continue
                    
                segments = list(load_records(segments_file))
                
                if len(found_beat_types_per_patient[patient]) == len(self.beat_types):
                    continue
                    
                for segment in tqdm(segments, desc=f"Processing {patient}"):
                    segment_record = os.path.join(patient_record, segment)
                    
                    if not os.path.exists(segment_record + ".atr"):
                        continue
                        
                    # Skip segments if we've already found all beat types for this patient
                    if len(found_beat_types_per_patient[patient]) == len(self.beat_types):
                        break
                        
                    try:
                        # Only search for beat types we haven't found yet
                        remaining_beat_types = [bt for bt in self.beat_types if bt not in found_beat_types_per_patient[patient]]
                        
                        ann = wfdb.rdann(segment_record, "atr")
                        # header = wfdb.rdheader(segment_record)
                        # sig_len = header.sig_len
                        sig_len = ann.sample[-1]
                        
                        # Find indices only for beat types we haven't found yet
                        for beat_type in remaining_beat_types:
                            indices = [i for i, s in enumerate(ann.symbol) if s == beat_type]
                            
                            if indices:  
                                found_beat_types_per_patient[patient].add(beat_type)
                                
                                random_idx = random.choice(indices)
                                sample = ann.sample[random_idx]
                                
                                start = max(0, sample - self.input_len // 2)
                                end = min(sig_len, sample + self.input_len // 2)
                                
                                if end - start == self.input_len:
                                    samples.append({
                                        'patient': patient,
                                        'segment': segment,
                                        'sample': sample,
                                        'signal_path': segment_record,
                                        'start': start,
                                        'end': end,
                                        'label': self.beat_map[beat_type]
                                    })
                    except Exception as e:
                        print(f"Error processing {segment_record}: {e}")
            
            # Print statistics about the collected samples
            beat_type_counts = Counter([self.beat_types[s['label']] for s in samples])
            print(f"Collected samples by beat type: {dict(beat_type_counts)}")
            print(f"Total samples: {len(samples)}")
                        
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        signal = wfdb.rdrecord(sample['signal_path'], sampfrom=sample['start'], sampto=sample['end'])

        ecg_signal = signal.p_signal
        ecg_signal = bandpass_filter(ecg_signal.transpose(), 0.67, 40, fs=250)
        ecg_signal = processing.resample_sig(ecg_signal[0], 250, 500)[0]
        ecg_signal = np.expand_dims(ecg_signal, axis=1)
        ecg_signal = z_score_normalization(ecg_signal)
        
        ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        if self.transform:
            ecg_signal = self.transform(ecg_signal)
            
        return ecg_signal.transpose(1, 0), label


class IcentiaRhythmDataset(Dataset):
    def __init__(self, data_path, patient_ids, stage="train", input_len=1000, transform=None):
        """
        Dataset for ECG rhythm classification
        
        Args:
            data_path: Root directory containing ECG records
            patient_ids: List of patient IDs to include
            input_len: Length of ECG segment to extract
            transform: Optional transforms to apply
        """
        self.data_path = data_path
        self.patient_ids = patient_ids
        self.stage = stage  # test, val, train
        self.input_len = input_len
        self.transform = transform
        
        # Rhythm labels: N (normal), AFIB (atrial fibrillation), AFL (atrial flutter)
        self.rhythm_types = ['N', 'AFIB', 'AFL']
        self.rhythm_map = {rhythm: idx for idx, rhythm in enumerate(self.rhythm_types)}
        
        # Prepare samples
        self.samples = self._prepare_samples()
        
        # Calculate class weights
        labels = [s['label'] for s in self.samples]
        class_counts = Counter(labels)
        total_samples = len(labels)
        self.class_weight = {cls: total_samples / count for cls, count in class_counts.items()}
        
    def _prepare_samples(self):
        if os.path.exists(f"dataset/Icentia11K/rhythm/icentia_rhythm_{self.stage}.json"):
            # If the dataset is already prepared, load it
            with open(f"dataset/Icentia11K/rhythm/icentia_rhythm_{self.stage}.json", 'r') as f:
                samples = json.load(f)
            print(f"Loaded {len(samples)} samples from preprocessed dataset.")
            
        else:
            samples = []
            
            # Track which beat types have been found for each patient
            found_beat_types_per_patient = {patient: set() for patient in self.patient_ids}
            
            for patient in self.patient_ids:
                patient_record = f'{self.data_path}/{patient}'[:-1]
                
                segments_file = os.path.join(patient_record, "RECORDS")
                if not os.path.exists(segments_file):
                    continue
                    
                segments = list(load_records(segments_file))
                
                if len(found_beat_types_per_patient[patient]) == len(self.beat_types):
                    continue
                    
                for segment in tqdm(segments, desc=f"Processing {patient}"):
                    segment_record = os.path.join(patient_record, segment)
                    
                    if not os.path.exists(segment_record + ".atr"):
                        continue
                        
                    # Skip segments if we've already found all beat types for this patient
                    if len(found_beat_types_per_patient[patient]) == len(self.beat_types):
                        break
                        
                    try:
                        # Only search for beat types we haven't found yet
                        remaining_beat_types = [bt for bt in self.beat_types if bt not in found_beat_types_per_patient[patient]]
                        
                        ann = wfdb.rdann(segment_record, "atr")
                        # header = wfdb.rdheader(segment_record)
                        # sig_len = header.sig_len
                        sig_len = ann.sample[-1]
                        
                        # Find indices only for beat types we haven't found yet
                        for beat_type in remaining_beat_types:
                            indices = [i for i, s in enumerate(ann.aux_note) if s == beat_type]
                            
                            if indices:  
                                found_beat_types_per_patient[patient].add(beat_type)
                                
                                random_idx = random.choice(indices)
                                sample = ann.sample[random_idx]
                                
                                start = max(0, sample - self.input_len // 2)
                                end = min(sig_len, sample + self.input_len // 2)
                                
                                if end - start == self.input_len:
                                    samples.append({
                                        'patient': patient,
                                        'segment': segment,
                                        'sample': sample,
                                        'signal_path': segment_record,
                                        'start': start,
                                        'end': end,
                                        'label': self.beat_map[beat_type]
                                    })
                    except Exception as e:
                        print(f"Error processing {segment_record}: {e}")
            
            # Print statistics about the collected samples
            beat_type_counts = Counter([self.beat_types[s['label']] for s in samples])
            print(f"Collected samples by beat type: {dict(beat_type_counts)}")
            print(f"Total samples: {len(samples)}")
                        
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        signal = wfdb.rdrecord(sample['signal_path'], sampfrom=sample['start'], sampto=sample['end'])
        
        ecg_signal = signal.p_signal
        ecg_signal = bandpass_filter(ecg_signal.transpose(), 0.67, 40, fs=250)
        ecg_signal = processing.resample_sig(ecg_signal[0], 250, 500)[0]
        ecg_signal = np.expand_dims(ecg_signal, axis=1)
        ecg_signal = z_score_normalization(ecg_signal)
        
        ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        if self.transform:
            ecg_signal = self.transform(ecg_signal)
            
        return ecg_signal.transpose(1, 0), label


class PTBXLDataset(Dataset):
    def __init__(self, data, labels, input_len=1000):
        """
        PTB-XL Dataset for ECG classification

        Args:
            data: Numpy array of ECG signals
            labels: One-hot encoded labels
        """
        self.data = data
        self.labels = labels
        self.input_len = input_len
        self.class_count = np.sum(self.labels, axis=0)
        self.class_weight = self.class_count / np.sum(self.class_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data.shape[-1] == 12:
            resampled_leads = []
            for lead_idx in range(self.data.shape[2]):  # Iterate through all 12 leads
                resampled_lead = processing.resample_sig(self.data[idx][:, lead_idx], 100, 500)[0]
                resampled_leads.append(resampled_lead)
            data = np.stack(resampled_leads, axis=0)
            signal = torch.tensor(data, dtype=torch.float32)[
                : self.input_len, :
            ]
        else:
            data = processing.resample_sig(self.data[idx][:, 0], 100, 500)[0]
            data = np.expand_dims(data, axis=1)
            signal = torch.tensor(data, dtype=torch.float32)[
                : self.input_len, :
            ].transpose(0, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return signal, label


class MusicDataset(Dataset):
    def __init__(
        self,
        data_path,
        label_csv,
        task="classification",
        input_len=5000,
    ):
        self.data_path = data_path
        self.df = pd.read_csv(label_csv)
        self.task = task
        self.input_len = input_len
        self.num_slices = 1
        self.class_count = self.df["outcome"].value_counts().to_dict()
        self.class_weight = {
            k: v / sum(self.class_count.values()) for k, v in self.class_count.items()
        }

    def __len__(self):
        return len(self.df)

    def _load_signal(self, subj_id, start, end):
        file_path = os.path.join(self.data_path, subj_id)
        record = wfdb.rdrecord(file_path, channels=[0], sampfrom=start, sampto=end)
        # return record.p_signal[:, 0]  # return as 1D np array
        return processing.resample_sig(record.p_signal[:, 0], record.fs, 500)[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subj_id = row["subj_id"]
        # Remove possible file extension for wfdb
        subj_id = subj_id.split(".")[0]
       
        start = 0
        end = self.input_len * 200 // 500
        max_attempts = 10  # Try up to 10 windows
        found_valid = False
        for _ in range(max_attempts):
            x_np = self._load_signal(subj_id, start, end)
            if is_valid_signal(x_np) and not is_flat_signal(x_np):
                found_valid = True
                break
            # Move window forward by input_len (non-overlapping)
            start += self.input_len
            end += self.input_len
        if not found_valid:
            # If no valid window found, just use the first window
            x_np = self._load_signal(subj_id, 0, self.input_len)

        # signal = np.expand_dims(x_np, axis=0) 
        # signal = z_score_normalization(signal)
        # signal = bandpass_filter(signal, 0.67, 40, fs=200)

        signal = torch.from_numpy(x_np).float().unsqueeze(0)  # shape: [1, input_len]
        
        if self.task == "classification":
            label = torch.tensor(int(row["outcome"]), dtype=torch.long)
            # label = F.one_hot(torch.tensor(int(row["outcome"]), dtype=torch.long), num_classes=4).float()
        else:
            label = torch.tensor(float(row["follow_up"]), dtype=torch.float)
        return signal, label


class LVEF_12lead_cls_Dataset(Dataset):
    def __init__(self, ecg_path, labels_df, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the annotations.
            data_dir (str): Directory path containing the numpy data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
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
        self.new_leads = [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        result = 0
        hash_file_name = str(self.labels_df.iloc[idx, 1])
        labels = self.labels_df.iloc[idx, -1]
        labels = labels.astype(np.float32)
        data = [wfdb.rdsamp(self.ecg_path + hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        result = self.check_nan_in_array(data)
        if result != 0:
            print(hash_file_name)
        data = data.squeeze(0)
        data = np.transpose(data, (1, 0))
        data = data[self.lead_indices, :]
        signal = z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        # Convert to torch tensors
        labels = torch.tensor(labels, dtype=torch.float)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1:
            labels = labels.unsqueeze(1)
        return signal, labels


class LVEF_12lead_reg_Dataset(Dataset):
    def __init__(self, ecg_path, labels_df, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the annotations.
            data_dir (str): Directory path containing the numpy data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
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
        self.new_leads = [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        hash_file_name = str(self.labels_df.iloc[idx, 1])
        labels = self.labels_df.iloc[idx, -2]
        labels = torch.tensor(
            [labels], dtype=torch.float32
        )  # Wrap the label in a list to create an extra dimension
        data = [wfdb.rdsamp(self.ecg_path + hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        data = data.squeeze(0)
        data = np.transpose(data, (1, 0))
        data = data[self.lead_indices, :]
        signal = z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        return signal, labels


class LVEF_1lead_cls_Dataset(Dataset):
    def __init__(self, ecg_path, labels_df, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the annotations.
            data_dir (str): Directory path containing the numpy data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
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
        self.new_leads = ["II"]
        # self.new_leads = ["I"]
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        result = 0
        hash_file_name = str(self.labels_df.iloc[idx, 1])
        labels = self.labels_df.iloc[idx, -1]
        labels = labels.astype(np.float32)
        data = [wfdb.rdsamp(self.ecg_path + hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        result = self.check_nan_in_array(data)
        if result != 0:
            print(hash_file_name)
        data = data.squeeze(0)
        data = np.transpose(data, (1, 0))
        data = data[self.lead_indices, :]
        # signal = bandpass_filter(data, 0.67, 40, fs=500)
        signal = z_score_normalization(data)
        signal = torch.FloatTensor(signal)
        # signal = torch.FloatTensor(signal)[:,:2000]

        # Convert to torch tensors
        labels = torch.tensor(labels, dtype=torch.float)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1:
            labels = labels.unsqueeze(1)
        return signal, labels


class LVEF_1lead_reg_Dataset(Dataset):
    def __init__(self, ecg_path, labels_df, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the annotations.
            data_dir (str): Directory path containing the numpy data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
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
        self.new_leads = ["II"]
        # self.new_leads = ["I"]
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        hash_file_name = str(self.labels_df.iloc[idx, 1])
        labels = self.labels_df.iloc[idx, -2]
        labels = torch.tensor(
            [labels], dtype=torch.float32
        )  # Wrap the label in a list to create an extra dimension
        data = [wfdb.rdsamp(self.ecg_path + hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        data = data.squeeze(0)
        data = np.transpose(data, (1, 0))
        data = data[self.lead_indices, :]
        signal = z_score_normalization(data)
        signal = torch.FloatTensor(signal)
        # signal = torch.FloatTensor(signal)[:,:2000]

        return signal, labels


class MCMEDECGDataset(Dataset):
    def __init__(self, data_path, label_df, stage="train", input_len=2500, task="classification", 
                 target_fs=500, normalize=True, label_column="Triage_acuity"):
        """
        MCMED Dataset for clinical monitoring ECG data
        
        Args:
            data_path: Path to MCMED root directory containing data folder
            label_path: Path to variables CSV file containing labels
            stage: 'train', 'val', or 'test'
            input_len: Length of ECG segment (samples)
            task: 'classification' or 'regression'
            target_fs: Target sampling frequency
            normalize: Whether to apply z-score normalization
            label_column: Column name in variables CSV to use as labels
        """
        self.data_path = data_path
        self.stage = stage
        self.input_len = input_len
        self.task = task
        self.target_fs = target_fs
        self.normalize = normalize
        self.label_column = label_column
        
        # Load split file
        split_file = os.path.join(data_path, f"split_chrono_{stage}.csv")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        self.split_df = pd.read_csv(split_file, header=None)
        print(f"Loaded {len(self.split_df)} patients for {stage} set")
        
        self.variables_df = label_df
        
        self.label_map = {}
        self.ii_segments_map = {}
        
        for _, row in self.variables_df.iterrows():
            enc_id = str(row['visit_id'])

            if label_column in row and not pd.isna(row[label_column]):
                if task == "classification":
                    raw_label = str(row[label_column])
                    if label_column == "Triage_acuity":
                        # Convert "2-Emergent" -> 2, "3-Urgent" -> 3, etc.  
                        acuity_level = int(raw_label.split('-')[0])
                        self.label_map[enc_id] = acuity_level - 1  # Convert to 0-indexed
                        
                    elif self.label_column == "ED_dispo":
                        # ED disposition mapping
                        ed_dispo_map = {
                            "Discharge": 0,
                            "Inpatient": 1,
                            "Observation": 2,
                            "ICU": 3
                        }
                        self.label_map[enc_id] = ed_dispo_map.get(str(raw_label), 0)
                        
                    elif self.label_column == "DC_dispo":
                        # DC disposition mapping - group similar dispositions
                        dc_dispo_str = str(raw_label)
                        
                        # Home/Self-Care (Class 0)
                        if any(keyword in dc_dispo_str for keyword in [
                            "Home/Work", "Left Against Medical Advice"
                        ]):
                            self.label_map[enc_id] = 0
                            
                        # Care Facility (Class 1)
                        elif any(keyword in dc_dispo_str for keyword in [
                            "Home Health Services", "Skilled Nursing Facility", 
                            "Residential Care Facility", "Custodial or Supportive Care Facility",
                            "Skilled Nursing Facility- Planned Inpatient Readmit",
                            "ValleyCare Skilled Nursing Facility"
                        ]):
                            self.label_map[enc_id] = 1
                            
                        # Hospital Transfer (Class 2)
                        elif any(keyword in dc_dispo_str for keyword in [
                            "Acute Care Hospital", "Long Term Care Hospital", 
                            "Federal Hospital", "Lucile Packard Children's Hospital",
                            "Stanford Children's Hospital", "Critical Access Hospital",
                            "Planned inpatient readmission", "Stanford Acute Care"
                        ]):
                            self.label_map[enc_id] = 2
                            
                        # Hospice/End-of-Life (Class 3)
                        elif any(keyword in dc_dispo_str for keyword in [
                            "Expired", "Hospice", "Organ Harvest", "Donor"
                        ]):
                            self.label_map[enc_id] = 3
                            
                        # Psychiatric/Rehabilitation (Class 4)
                        elif any(keyword in dc_dispo_str for keyword in [
                            "Psychiatric Hospital", "Stanford Psych", 
                            "Inpatient Rehabilitation Facility"
                        ]):
                            self.label_map[enc_id] = 4
                            
                        # Other/Unknown (Class 5)
                        else:
                            self.label_map[enc_id] = 5
                    else:
                        raise ValueError(f"Unsupported label column for classification: {label_column}")
                else:  # regression
                    if self.label_column in ["Triage_SBP", "Triage_DBP"]:
                        self.label_map[enc_id] = float(row[label_column])
                    else:
                        raise ValueError(f"Unsupported label column for regression: {label_column}")
            
            # Store II segments info
            if 'II_n_segments' in row and not pd.isna(row['II_n_segments']):
                self.ii_segments_map[enc_id] = int(row['II_n_segments'])
        
        print(f"Found labels for {len(self.label_map)} encounters")
        print(f"Found II segment info for {len(self.ii_segments_map)} encounters")
        
        if os.path.exists(f"dataset/mcmed/data_{self.label_column}_{self.stage}.csv"):
            # If the dataset is already prepared, load it
            with open(f"dataset/mcmed/data_{self.label_column}_{self.stage}.csv", 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.samples = [row for row in reader]
            print(f"Loaded {len(self.samples)} samples from preprocessed dataset.")
        else:
            self.samples = self._find_ecg_records()
        
        if task == "classification" and len(self.samples) > 0:
            labels = [sample['label'] for sample in self.samples]
            label_counts = Counter(labels)
            print(f"Label distribution: {dict(label_counts)}")
        
        print(f"{stage.upper()} set: {len(self.samples)} ECG records from "
              f"{len(self.split_df)} patients with valid labels")
    
    def _find_ecg_records(self):
        """Find available ECG records for patients in the split that have labels"""
        samples = []
        waveforms_dir = os.path.join(self.data_path, 'waveforms')
        
        for _, row in self.split_df.iterrows():
            enc_id = str(row.iloc[0])
            patient_id = enc_id[-3:]  # Last 3 digits CSN
            
            if enc_id not in self.label_map:
                continue
                
            if enc_id not in self.ii_segments_map or self.ii_segments_map[enc_id] == 0:
                continue
            
            patient_dir = os.path.join(waveforms_dir, patient_id)
            if not os.path.exists(patient_dir):
                continue
            
            record_path = os.path.join(patient_dir, enc_id)
            lead_ii_dir = os.path.join(record_path, 'II')
            
            if not os.path.exists(lead_ii_dir):
                continue

            # Look for ECG files
            dat_file = os.path.join(lead_ii_dir, f"{enc_id}_1.dat")
            hea_file = os.path.join(lead_ii_dir, f"{enc_id}_1.hea")
            
            if os.path.exists(dat_file) and os.path.exists(hea_file):
                samples.append({
                    'label': self.label_map[enc_id],
                    'record_path': os.path.join(lead_ii_dir, f"{enc_id}_1")
                })
        
        with open(f"dataset/mcmed/data_{self.label_column}_{self.stage}.csv", 'w', newline='', encoding='utf-8') as f:
            if samples:
                fieldnames = ['dat_file', 'hea_file', 'label', 'record_path']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(samples)

        return samples
    
    def _load_ecg_signal(self, record_path):
        """Load ECG signal from WFDB format"""
        try:
            record = wfdb.rdrecord(record_path, sampfrom=0, sampto=5000)
            
            if len(record.p_signal.shape) > 1:
                ecg_signal = record.p_signal[:, 0]  # Lead II
            else:
                ecg_signal = record.p_signal

            return ecg_signal
            
        except Exception as e:
            print(f"Error loading {record_path}: {e}")
            return np.zeros(self.input_len)
    
    def _process_signal(self, signal):
        """Process ECG signal to fixed length with optional filtering and normalization"""
        # Handle signal length
        if len(signal) >= self.input_len:
            # Random crop if signal is longer
            start_idx = np.random.randint(0, len(signal) - self.input_len + 1)
            signal = signal[start_idx:start_idx + self.input_len]
        else:
            # Pad with zeros if signal is shorter
            signal = np.pad(signal, (0, self.input_len - len(signal)), 'constant')
        
        # Apply bandpass filter (0.67-40 Hz)
        try:
            signal_filtered = bandpass_filter(
                signal.reshape(1, -1), 0.67, 40, fs=self.target_fs
            )[0]
        except Exception as e:
            print(f"Filtering failed: {e}, using unfiltered signal")
            signal_filtered = signal
        
        # Z-score normalization
        if self.normalize and np.std(signal_filtered) > 0:
            signal_filtered = z_score_normalization(signal_filtered)
        
        return signal_filtered
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        ecg_signal = self._load_ecg_signal(sample['record_path'])
        ecg_signal = self._process_signal(ecg_signal)
        
        ecg_tensor = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0)  # [1, input_len]
        
        # Get the actual label for this sample
        if self.task == "classification":
            label = torch.tensor(int(sample['label']), dtype=torch.long)
        else:  # regression
            sample_id = sample["record_path"].split('/')[-1].split('_')[0]
            # label = torch.tensor(float(sample['label']), dtype=torch.float32)
            label = self.label_map[sample_id]

        return ecg_tensor, label
    
    def get_class_distribution(self):
        """Get distribution of classes for classification tasks"""
        if self.task != "classification":
            return None
            
        labels = [sample['label'] for sample in self.samples]
        label_counts = Counter(labels)
        return {
            'counts': dict(label_counts),
            'total': len(labels)
        }
    
    def get_label_info(self):
        """Get information about the labels"""
        if self.task == "classification":
            if self.label_column == "Triage_acuity":
                # Triage_acuity
                return {
                    "type": "classification",
                    "num_classes": 5,
                    "class_names": {
                        0: "1-Resuscitation", # 875
                        1: "2-Emergent", # 22179
                        2: "3-Urgent", # 56529
                        3: "4-Semi-Urgent", # 3505
                        4: "5-Non-Urgent" # 150
                    }
                }
            elif self.label_column == "ED_dispo":
                return {
                    "type": "classification",
                    "num_classes": 4,
                    "class_names": {
                        0: "Discharge", # 45119
                        1: "Inpatient", # 23576
                        2: "Observation", # 12626
                        3: "ICU" # 2269
                    }
                }
            elif self.label_column == "DC_dispo":
                return {
                    "type": "classification",
                    "num_classes": 6,
                    "class_names": {
                        0: "Home/Self-Care",  # 69001 + 506 (Home/Work + Left AMA)
                        1: "Care Facility",   # 4707 + 3347 + 96 + 12 + 11 (Home Health + SNF + Residential + Custodial + SNF Readmit)
                        2: "Hospital Transfer", # 164 + 44 + 42 + 10 + 7 + 5 + 1 (Various hospital transfers)
                        3: "Hospice/End-of-Life", # 659 + 604 + 117 + 6 + 2 (Expired + Hospice facilities + Organ donation)
                        4: "Psychiatric/Rehabilitation", # 43 + 21 + 281 + 3 + 34 (Psychiatric + Inpatient Rehab facilities)
                        5: "Other/Unknown"    # 3704 + 68 + 55 + 15 + 3 + 1 + 1 + 1 (No disposition + Other + Jail + Various others)
                    }
                }
        else:
            return {
                "type": "regression",
                "target": self.label_column
            }
    

